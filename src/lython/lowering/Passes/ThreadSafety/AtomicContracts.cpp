#include "Verifier.h"

#include "Common/ClassLayout.h"
#include "Common/Object.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/ADT/Twine.h"

namespace py::threadsafe {

mlir::LogicalResult
verifier::llvm::LockAcquire::controlFlow(mlir::LLVM::AtomicRMWOp op) {
  mlir::Block *lockBlock = op->getBlock();
  for (mlir::Operation *resultUser : op->getResult(0).getUsers()) {
    auto cmp = mlir::dyn_cast<mlir::LLVM::ICmpOp>(resultUser);
    if (!cmp)
      continue;
    bool trueMeansAcquired =
        compare::llvmZero(cmp, op->getResult(0), mlir::LLVM::ICmpPredicate::eq);
    bool falseMeansAcquired =
        compare::llvmZero(cmp, op->getResult(0), mlir::LLVM::ICmpPredicate::ne);
    if (!trueMeansAcquired && !falseMeansAcquired)
      continue;

    for (mlir::Operation *cmpUser : cmp->getResult(0).getUsers()) {
      auto branch = mlir::dyn_cast<mlir::LLVM::CondBrOp>(cmpUser);
      if (!branch || branch.getCondition() != cmp->getResult(0) ||
          branch->getNumSuccessors() < 2)
        continue;
      mlir::Block *trueDest = branch->getSuccessor(0);
      mlir::Block *falseDest = branch->getSuccessor(1);
      if (trueMeansAcquired && trueDest != lockBlock && falseDest == lockBlock)
        return mlir::success();
      if (falseMeansAcquired && falseDest != lockBlock && trueDest == lockBlock)
        return mlir::success();
    }
  }
  return op->emitOpError("lock acquire result must branch to the critical "
                         "section only when the previous lock value is zero");
}

bool retain_op::runtimeCall(mlir::Operation *op) {
  auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(op);
  if (!call)
    return false;
  if (!attrs::i64Array(call.getOperation(), OwnershipContractAttrs::kRetainArgs)
           .empty())
    return true;
  return false;
}

bool retain_op::atomic(mlir::Operation *op) {
  auto role = attrs::str(op, ThreadSafetyAttrs::kAtomicRole);
  return role && (*role == ThreadSafetyAttrs::kRoleContainerRefcountRetain ||
                  *role == ThreadSafetyAttrs::kRoleClassRefcountRetain ||
                  *role == ThreadSafetyAttrs::kRoleObjectRefcountRetain);
}

namespace aggregate {

static bool borrow(mlir::Value value,
                   ::llvm::SmallPtrSetImpl<mlir::Value> &seen) {
  if (!value || !seen.insert(value).second)
    return false;

  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *owner = arg.getOwner();
    for (mlir::Block *pred : owner->getPredecessors()) {
      mlir::Operation *terminator = pred->getTerminator();
      auto branch = mlir::dyn_cast_or_null<mlir::BranchOpInterface>(terminator);
      if (!branch)
        continue;
      for (unsigned i = 0, e = branch->getNumSuccessors(); i != e; ++i) {
        if (branch->getSuccessor(i) != owner)
          continue;
        mlir::SuccessorOperands operands = branch.getSuccessorOperands(i);
        unsigned argNo = arg.getArgNumber();
        if (argNo >= operands.size() || operands.isOperandProduced(argNo))
          continue;
        if (borrow(operands[argNo], seen))
          return true;
      }
    }
  }

  if (auto bitcast = value.getDefiningOp<mlir::LLVM::BitcastOp>())
    return borrow(bitcast.getArg(), seen);
  if (auto intToPtr = value.getDefiningOp<mlir::LLVM::IntToPtrOp>())
    return borrow(intToPtr.getArg(), seen);
  if (auto ptrToInt = value.getDefiningOp<mlir::LLVM::PtrToIntOp>())
    return borrow(ptrToInt.getArg(), seen);

  if (auto gep = value.getDefiningOp<mlir::LLVM::GEPOp>()) {
    if (gep->hasAttr(OwnershipContractAttrs::kAggregateSlotLoad))
      return true;
    return borrow(gep.getBase(), seen);
  }
  if (auto extract = value.getDefiningOp<mlir::LLVM::ExtractValueOp>()) {
    if (extract->hasAttr(OwnershipContractAttrs::kAggregateSlotLoad))
      return true;
    return borrow(extract.getContainer(), seen);
  }
  if (auto insert = value.getDefiningOp<mlir::LLVM::InsertValueOp>()) {
    if (insert->hasAttr(OwnershipContractAttrs::kAggregateSlotLoad))
      return true;
    return borrow(insert.getValue(), seen) ||
           borrow(insert.getContainer(), seen);
  }

  if (auto load = value.getDefiningOp<mlir::LLVM::LoadOp>()) {
    auto role = attrs::str(load.getOperation(), ThreadSafetyAttrs::kAtomicRole);
    if (role && *role == ThreadSafetyAttrs::kRoleAsyncExceptionLoad)
      return true;
    return load->hasAttr(OwnershipContractAttrs::kAggregateSlotLoad);
  }
  if (auto load = value.getDefiningOp<mlir::memref::LoadOp>())
    return load->hasAttr(OwnershipContractAttrs::kAggregateSlotLoad);

  return false;
}

} // namespace aggregate

namespace retain {

static mlir::Value target(mlir::Operation *op) {
  if (auto atomic = mlir::dyn_cast<mlir::memref::AtomicRMWOp>(op))
    return atomic.getMemref();
  if (auto atomic = mlir::dyn_cast<mlir::memref::GenericAtomicRMWOp>(op))
    return atomic.getMemref();
  if (auto atomic = mlir::dyn_cast<mlir::LLVM::AtomicRMWOp>(op))
    return atomic.getPtr();
  if (auto atomic = mlir::dyn_cast<mlir::LLVM::AtomicCmpXchgOp>(op))
    return atomic.getPtr();
  if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(op))
    if (call.getNumOperands() >= 1)
      return call.getOperand(0);
  return {};
}

static bool entryBorrowed(mlir::Operation *op) {
  mlir::Value value = target(op);
  return value && provenance::entryArgRoot(value);
}

static bool aggregateBorrow(mlir::Operation *op) {
  if (op->hasAttr(OwnershipContractAttrs::kAggregateSlotLoad) &&
      op->hasAttr(OwnershipContractAttrs::kAggregateSlotGroup) &&
      op->hasAttr(OwnershipContractAttrs::kAggregateSlotComponent))
    return true;
  if (op->hasAttr(ThreadSafetyAttrs::kAtomicMemRefGroup) &&
      op->hasAttr(ThreadSafetyAttrs::kAtomicMemRefComponent))
    return true;
  mlir::Value value = target(op);
  ::llvm::SmallPtrSet<mlir::Value, 8> seen;
  return aggregate::borrow(value, seen);
}

} // namespace retain

bool dominance::block(mlir::Block *block, mlir::Operation *op,
                      mlir::DominanceInfo &dominance) {
  if (!block || !op)
    return false;
  if (op->getBlock() == block)
    return true;
  return dominance.dominates(block, op->getBlock());
}

namespace helper {

static bool branchDominates(mlir::Operation *op, mlir::LLVM::LLVMFuncOp fn,
                            unsigned argIndex, bool trueEdge) {
  if (!fn || fn.getBody().empty())
    return false;
  mlir::Block &entry = fn.getBody().front();
  if (argIndex >= entry.getNumArguments())
    return false;

  mlir::Value guard = entry.getArgument(argIndex);
  mlir::DominanceInfo dominance(fn.getOperation());
  bool found = false;
  fn.walk([&](mlir::LLVM::CondBrOp branch) {
    if (found || branch.getCondition() != guard ||
        branch->getNumSuccessors() < 2)
      return;
    mlir::Block *successor = branch->getSuccessor(trueEdge ? 0 : 1);
    found = ::py::threadsafe::dominance::block(successor, op, dominance);
  });
  return found;
}

static int64_t findI1Arg(mlir::LLVM::LLVMFuncOp fn, bool first) {
  if (!fn || fn.getBody().empty())
    return -1;
  mlir::Block &entry = fn.getBody().front();
  int64_t found = -1;
  for (auto [index, arg] : llvm::enumerate(entry.getArguments())) {
    auto integer = mlir::dyn_cast<mlir::IntegerType>(arg.getType());
    if (!integer || integer.getWidth() != 1)
      continue;
    found = static_cast<int64_t>(index);
    if (first)
      return found;
  }
  return found;
}

static bool branchDominatesI1(mlir::Operation *op, mlir::LLVM::LLVMFuncOp fn,
                              bool first, bool trueEdge) {
  int64_t argIndex = findI1Arg(fn, first);
  if (argIndex < 0)
    return false;
  return branchDominates(op, fn, static_cast<unsigned>(argIndex), trueEdge);
}

enum class Guard { none, firstI1True, lastI1False };

struct OwnedTokenContext {
  ::llvm::StringRef role;
  ::llvm::StringRef helperKind;
  Guard guard;
  ::llvm::StringRef guardError;
};

static constexpr OwnedTokenContext kOwnedTokenContexts[] = {
    {ThreadSafetyAttrs::kRoleClassRefcountRetain, ClassSafetyAttrs::kKindIncref,
     Guard::none, ""},
    {ThreadSafetyAttrs::kRoleContainerRefcountRetain,
     ClassSafetyAttrs::kKindGetField, Guard::lastI1False,
     "getfield helper owned-token retain is not dominated by the local-retain "
     "branch"},
    {ThreadSafetyAttrs::kRoleContainerRefcountRetain,
     ClassSafetyAttrs::kKindSetField, Guard::firstI1True,
     "setfield helper owned-token retain is not dominated by the "
     "retain-new-value branch"},
};

static const OwnedTokenContext *
lookupOwnedTokenContext(::llvm::StringRef role, ::llvm::StringRef helperKind) {
  for (const OwnedTokenContext &context : kOwnedTokenContexts)
    if (role == context.role && helperKind == context.helperKind)
      return &context;
  return nullptr;
}

static bool guardDominates(mlir::Operation *op, mlir::LLVM::LLVMFuncOp fn,
                           Guard guard) {
  switch (guard) {
  case Guard::none:
    return true;
  case Guard::firstI1True:
    return branchDominatesI1(op, fn, /*first=*/true, /*trueEdge=*/true);
  case Guard::lastI1False:
    return branchDominatesI1(op, fn, /*first=*/false, /*trueEdge=*/false);
  }
  return false;
}

static mlir::LogicalResult ownedTokenRetainContext(mlir::Operation *op) {
  auto fn = op->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  if (!fn)
    return op->emitOpError("class-helper owned-token retain is outside an "
                           "LLVM helper function");

  auto role = attrs::str(op, ThreadSafetyAttrs::kAtomicRole);
  if (!role)
    return op->emitOpError("class-helper owned-token retain is missing role");

  auto helperKind =
      attrs::str(fn.getOperation(), ClassSafetyAttrs::kHelperKind);
  auto helperClass =
      attrs::str(fn.getOperation(), ClassSafetyAttrs::kHelperClass);
  if (!helperKind || !helperClass)
    return op->emitOpError(
        "class-helper owned-token proof lacks helper metadata");

  const OwnedTokenContext *context =
      lookupOwnedTokenContext(*role, *helperKind);
  if (!context)
    return op->emitOpError(
        "class-helper owned-token proof has no role/helper context");
  if (guardDominates(op, fn, context->guard))
    return mlir::success();
  return op->emitOpError() << context->guardError;
}

} // namespace helper

mlir::LogicalResult
verifier::refcount::RetainPremise::verify(mlir::Operation *op) {
  if (!retain_op::atomic(op) && !retain_op::runtimeCall(op))
    return mlir::success();

  auto premise = attrs::str(op, ThreadSafetyAttrs::kRetainPremise);
  if (!premise)
    return op->emitOpError("shared retain is missing a retain premise");
  if (*premise == ThreadSafetyAttrs::kPremiseOwnedToken) {
    if (retain_op::atomic(op)) {
      if (!op->hasAttr(ThreadSafetyAttrs::kOwnedTokenVerified))
        return op->emitOpError("owned-token atomic retain lacks provenance "
                               "from the quantitative ownership verifier");
      auto proof = attrs::str(op, ThreadSafetyAttrs::kOwnedTokenProof);
      if (!proof)
        return op->emitOpError("owned-token atomic retain is missing proof "
                               "source provenance");
      if (*proof == ThreadSafetyAttrs::kProofOwnershipVerifier)
        return mlir::success();
      if (*proof == ThreadSafetyAttrs::kProofClassFieldHelper)
        return helper::ownedTokenRetainContext(op);
      return op->emitOpError("owned-token atomic retain has unsupported proof "
                             "source: ")
             << *proof;
    }
    return mlir::success();
  }
  if (*premise == ThreadSafetyAttrs::kPremiseEntryBorrowed) {
    if (retain::entryBorrowed(op))
      return mlir::success();
    return op->emitOpError("entry-borrowed retain premise does not target a "
                           "function entry argument resource");
  }
  if (*premise == ThreadSafetyAttrs::kPremiseCapturedBorrowed)
    return mlir::success();
  if (*premise == ThreadSafetyAttrs::kPremiseAggregateBorrow) {
    if (retain::aggregateBorrow(op))
      return mlir::success();
    return op->emitOpError("aggregate-borrow retain premise lacks aggregate "
                           "slot or exception-cell provenance");
  }
  if (*premise == ThreadSafetyAttrs::kPremiseLockedBorrow)
    return mlir::success();
  return op->emitOpError("shared retain has unsupported retain premise: ")
         << *premise;
}

namespace memref_index {

static mlir::LogicalResult single(mlir::memref::AtomicRMWOp op,
                                  int64_t expectedSlot,
                                  ::llvm::StringRef what) {
  if (op.getIndices().size() != 1)
    return op->emitOpError(what) << " must use exactly one header index";
  auto index = constant::index(op.getIndices().front());
  if (!index || *index != expectedSlot)
    return op->emitOpError(what) << " must target header slot " << expectedSlot;
  return mlir::success();
}

static mlir::LogicalResult single(mlir::memref::GenericAtomicRMWOp op,
                                  int64_t expectedSlot,
                                  ::llvm::StringRef what) {
  if (op.getIndices().size() != 1)
    return op->emitOpError(what) << " must use exactly one header index";
  auto index = constant::index(op.getIndices().front());
  if (!index || *index != expectedSlot)
    return op->emitOpError(what) << " must target header slot " << expectedSlot;
  return mlir::success();
}

} // namespace memref_index

namespace refcount_init {

static void collectEscapes(mlir::Value value,
                           ::llvm::SmallPtrSetImpl<mlir::Value> &seen,
                           mlir::SmallVectorImpl<mlir::Operation *> &escapes) {
  value = pointer::stripCasts(value);
  if (!value || !seen.insert(value).second)
    return;

  for (mlir::Operation *user : value.getUsers()) {
    if (local_container::escapeUser(user)) {
      escapes.push_back(user);
      continue;
    }
    if (local_container::use(user, value))
      continue;
    if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(user)) {
      collectEscapes(cast.getResult(), seen, escapes);
      continue;
    }
    escapes.push_back(user);
  }
}

static mlir::LogicalResult verifySealedBeforeEscape(mlir::memref::StoreOp store,
                                                    mlir::Value header) {
  mlir::Operation *scope = store->getParentOfType<mlir::func::FuncOp>();
  if (!scope)
    scope = store->getParentOfType<mlir::async::FuncOp>();
  if (!scope)
    scope = store->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  if (!scope)
    return store->emitOpError("managed container refcount initialization is "
                              "outside a verifiable function scope");

  mlir::SmallVector<mlir::Operation *> escapes;
  ::llvm::SmallPtrSet<mlir::Value, 8> seen;
  collectEscapes(header, seen, escapes);
  mlir::DominanceInfo dominance(scope);
  for (mlir::Operation *escape : escapes) {
    if (escape == store.getOperation())
      continue;
    if (!dominance.dominates(store.getOperation(), escape))
      return store->emitOpError(
          "managed container refcount must be initialized before the "
          "descriptor can escape");
  }
  return mlir::success();
}

static mlir::LogicalResult verifyManagedStore(mlir::memref::StoreOp store,
                                              int64_t initialRefcount) {
  auto init = store->getAttrOfType<mlir::IntegerAttr>(
      ContainerSafetyAttrs::kRefcountInit);
  if (!init || init.getInt() != initialRefcount)
    return store->emitOpError("managed container refcount initialization must "
                              "carry a matching positive refcount_init "
                              "contract");

  auto state =
      attrs::str(store.getOperation(), ContainerSafetyAttrs::kRefcountState);
  if (!state || *state != ContainerSafetyAttrs::kStateManaged)
    return store->emitOpError("managed container refcount initialization must "
                              "declare managed state");

  if (!memref_value::alloc(store.getMemref()))
    return store->emitOpError("managed container refcount initialization must "
                              "target a fresh container header");

  return verifySealedBeforeEscape(store, store.getMemref());
}

} // namespace refcount_init

namespace cmp {

static bool memrefZero(mlir::arith::CmpIOp cmp, mlir::Value value,
                       mlir::arith::CmpIPredicate expected) {
  if (cmp.getPredicate() != expected)
    return false;
  return (cmp.getLhs() == value && constant::memrefInt(cmp.getRhs(), 0)) ||
         (cmp.getRhs() == value && constant::memrefInt(cmp.getLhs(), 0));
}

} // namespace cmp

namespace lock {

static mlir::LogicalResult controlFlow(mlir::memref::AtomicRMWOp op) {
  mlir::Block *lockBlock = op->getBlock();
  for (mlir::Operation *resultUser : op->getResult(0).getUsers()) {
    auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(resultUser);
    if (!cmp)
      continue;
    bool trueMeansAcquired =
        cmp::memrefZero(cmp, op->getResult(0), mlir::arith::CmpIPredicate::eq);
    bool falseMeansAcquired =
        cmp::memrefZero(cmp, op->getResult(0), mlir::arith::CmpIPredicate::ne);
    if (!trueMeansAcquired && !falseMeansAcquired)
      continue;
    for (mlir::Operation *cmpUser : cmp->getResult(0).getUsers()) {
      if (auto condition = mlir::dyn_cast<mlir::scf::ConditionOp>(cmpUser)) {
        if (condition.getCondition() == cmp->getResult(0) &&
            condition->getParentOfType<mlir::scf::WhileOp>() &&
            falseMeansAcquired)
          return mlir::success();
        continue;
      }

      auto branch = mlir::dyn_cast<mlir::cf::CondBranchOp>(cmpUser);
      if (!branch || branch.getCondition() != cmp->getResult(0) ||
          branch->getNumSuccessors() < 2)
        continue;
      mlir::Block *trueDest = branch->getSuccessor(0);
      mlir::Block *falseDest = branch->getSuccessor(1);
      if (trueMeansAcquired && trueDest != lockBlock && falseDest == lockBlock)
        return mlir::success();
      if (falseMeansAcquired && falseDest != lockBlock && trueDest == lockBlock)
        return mlir::success();
    }
  }
  return op->emitOpError("lock acquire result must branch to the critical "
                         "section only when the previous lock value is zero");
}

} // namespace lock

namespace atomic_rmw {

enum class OrderingRule { Acquire, Release, AcqRel, RefcountInc };

enum class Target {
  AsyncExceptionLoad,
  ContainerRefcount,
  ObjectRefcount,
  ClassRefcount,
  ContainerLock,
  ClassLock,
  AsyncCancelRequest,
};

struct ContainerSlot {
  int64_t slot;
  ::llvm::StringRef kind;
};

struct Spec {
  ::llvm::StringLiteral role;
  Target target;
  int64_t value;
  OrderingRule ordering;
  ::llvm::StringLiteral what;
  bool retainPremise = false;
  bool acquireControlFlow = false;
};

static constexpr Spec kSpecs[] = {
    {ThreadSafetyAttrs::kRoleAsyncExceptionLoad, Target::AsyncExceptionLoad, 0,
     OrderingRule::Acquire, "async exception load"},
    {ThreadSafetyAttrs::kRoleContainerRefcountLoad, Target::ContainerRefcount,
     0, OrderingRule::Acquire, "container refcount load"},
    {ThreadSafetyAttrs::kRoleObjectRefcountLoad, Target::ObjectRefcount, 0,
     OrderingRule::Acquire, "object refcount load"},
    {ThreadSafetyAttrs::kRoleClassRefcountLoad, Target::ClassRefcount, 0,
     OrderingRule::Acquire, "class refcount load"},
    {ThreadSafetyAttrs::kRoleContainerRefcountRetain, Target::ContainerRefcount,
     1, OrderingRule::RefcountInc, "container retain", /*retainPremise=*/true},
    {ThreadSafetyAttrs::kRoleObjectRefcountRetain, Target::ObjectRefcount, 1,
     OrderingRule::RefcountInc, "object retain", /*retainPremise=*/true},
    {ThreadSafetyAttrs::kRoleContainerRefcountRelease,
     Target::ContainerRefcount, -1, OrderingRule::AcqRel, "container release"},
    {ThreadSafetyAttrs::kRoleObjectRefcountRelease, Target::ObjectRefcount, -1,
     OrderingRule::AcqRel, "object release"},
    {ThreadSafetyAttrs::kRoleContainerLockAcquire, Target::ContainerLock, 1,
     OrderingRule::Acquire, "container lock acquire",
     /*retainPremise=*/false, /*acquireControlFlow=*/true},
    {ThreadSafetyAttrs::kRoleContainerLockRelease, Target::ContainerLock, 0,
     OrderingRule::Release, "container lock release"},
    {ThreadSafetyAttrs::kRoleClassLockAcquire, Target::ClassLock, 1,
     OrderingRule::Acquire, "class lock acquire",
     /*retainPremise=*/false, /*acquireControlFlow=*/true},
    {ThreadSafetyAttrs::kRoleClassLockRelease, Target::ClassLock, 0,
     OrderingRule::Release, "class lock release"},
    {ThreadSafetyAttrs::kRoleAsyncCancelRequest, Target::AsyncCancelRequest, 1,
     OrderingRule::AcqRel, "async cancel request"},
};

static const Spec *lookup(::llvm::StringRef role) {
  for (const Spec &spec : kSpecs)
    if (role == spec.role)
      return &spec;
  return nullptr;
}

static bool accepts(OrderingRule rule, ::llvm::StringRef ordering) {
  switch (rule) {
  case OrderingRule::Acquire:
    return ordering::atLeastAcquire(ordering);
  case OrderingRule::Release:
    return ordering::atLeastRelease(ordering);
  case OrderingRule::AcqRel:
    return ordering::atLeastAcqRel(ordering);
  case OrderingRule::RefcountInc:
    return ordering::refcountInc(ordering);
  }
  return false;
}

static ::llvm::StringRef description(OrderingRule rule) {
  switch (rule) {
  case OrderingRule::Acquire:
    return "acquire or stronger";
  case OrderingRule::Release:
    return "release or stronger";
  case OrderingRule::AcqRel:
    return "acq_rel or stronger";
  case OrderingRule::RefcountInc:
    return "monotonic or stronger";
  }
  return "valid";
}

static mlir::LogicalResult verifyOrdering(mlir::Operation *op,
                                          ::llvm::StringRef actual,
                                          OrderingRule rule,
                                          ::llvm::StringRef what) {
  if (accepts(rule, actual))
    return mlir::success();
  return op->emitOpError() << what << " must be " << description(rule);
}

static mlir::LogicalResult verifyKindValue(mlir::memref::AtomicRMWOp op,
                                           mlir::arith::AtomicRMWKind kind,
                                           int64_t value,
                                           ::llvm::StringRef what,
                                           ::llvm::StringRef action) {
  if (op.getKind() == kind && constant::memrefInt(op.getValue(), value))
    return mlir::success();
  return op->emitOpError() << what << " must " << action;
}

static mlir::LogicalResult verifyLockMemRef(mlir::memref::AtomicRMWOp op,
                                            ::llvm::StringRef what) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(op.getMemRefType());
  auto intType =
      memrefType
          ? mlir::dyn_cast<mlir::IntegerType>(memrefType.getElementType())
          : mlir::IntegerType{};
  if (memrefType && memrefType.getRank() == 1 &&
      memrefType.getShape().front() == 1 && intType && intType.getWidth() == 32)
    return mlir::success();
  return op->emitOpError() << what << " must target memref<1xi32>";
}

static mlir::FailureOr<ContainerSlot>
resolveContainerSlot(mlir::memref::AtomicRMWOp op, bool lock,
                     ::llvm::StringRef what) {
  auto slot = lock ? header_slot::lock(op.getMemref())
                   : header_slot::refcount(op.getMemref());
  if (!slot)
    return op->emitOpError() << what << " must target a proven typed-container "
                             << (lock ? "lock component" : "header");

  auto containerKind = descriptor::Kind::get(op.getMemref());
  auto expected = containerKind
                      ? (lock ? header_slot::expectedLock(*containerKind)
                              : header_slot::expectedRefcount(*containerKind))
                      : std::optional<int64_t>{};
  if (!containerKind || !expected || *expected != *slot)
    return op->emitOpError()
           << what << " lacks exact "
           << (lock ? "lock component" : "typed header") << " kind provenance";

  if (mlir::failed(memref_index::single(op, *slot, what)))
    return mlir::failure();
  return ContainerSlot{*slot, *containerKind};
}

static mlir::LogicalResult
verifyAsyncExceptionLoad(mlir::memref::AtomicRMWOp op,
                         ::llvm::StringRef ordering) {
  if (!::py::async_runtime::isExceptionCellType(op.getMemRefType()))
    return op->emitOpError(
        "async exception load must target an exception-cell memref");
  if (mlir::failed(memref_index::single(op, 0, "async exception load")) ||
      mlir::failed(verifyKindValue(op, mlir::arith::AtomicRMWKind::addi, 0,
                                   "async exception load", "be addi 0")) ||
      mlir::failed(verifyOrdering(op.getOperation(), ordering,
                                  OrderingRule::Acquire,
                                  "async exception load")))
    return mlir::failure();
  ::py::async_runtime::ExceptionCell::mark(op.getMemref());
  return mlir::success();
}

static mlir::LogicalResult verifyContainerRefcount(mlir::memref::AtomicRMWOp op,
                                                   ::llvm::StringRef ordering,
                                                   int64_t delta,
                                                   OrderingRule orderingRule,
                                                   ::llvm::StringRef what,
                                                   bool verifyRetainPremise) {
  mlir::FailureOr<ContainerSlot> slot =
      resolveContainerSlot(op, /*lock=*/false, what);
  if (mlir::failed(slot))
    return mlir::failure();
  if (delta != 0 && memref_value::alloca(op.getMemref()))
    return op->emitOpError()
           << what << " must not mutate a local refcount-zero alloca header";
  if (verifyRetainPremise &&
      mlir::failed(
          verifier::refcount::RetainPremise::verify(op.getOperation())))
    return mlir::failure();

  std::string action = (::llvm::Twine("be addi ") + ::llvm::Twine(delta)).str();
  if (mlir::failed(verifyKindValue(op, mlir::arith::AtomicRMWKind::addi, delta,
                                   what, action)) ||
      mlir::failed(
          verifyOrdering(op.getOperation(), ordering, orderingRule, what)))
    return mlir::failure();
  resource::sealAtomic(op.getOperation(), op.getMemref(), slot->slot);
  return mlir::success();
}

static mlir::LogicalResult
verifyObjectRefcount(mlir::memref::AtomicRMWOp op, ::llvm::StringRef ordering,
                     int64_t slot, int64_t delta, OrderingRule orderingRule,
                     ::llvm::StringRef what, bool verifyRetainPremise) {
  if (!object_header::provenance(op.getMemref()))
    return op->emitOpError()
           << what << " must target a proven object header memref";
  if (mlir::failed(memref_index::single(op, slot, what)))
    return mlir::failure();
  if (verifyRetainPremise &&
      mlir::failed(
          verifier::refcount::RetainPremise::verify(op.getOperation())))
    return mlir::failure();

  std::string action = (::llvm::Twine("be addi ") + ::llvm::Twine(delta)).str();
  if (mlir::failed(verifyKindValue(op, mlir::arith::AtomicRMWKind::addi, delta,
                                   what, action)) ||
      mlir::failed(
          verifyOrdering(op.getOperation(), ordering, orderingRule, what)))
    return mlir::failure();
  resource::sealAtomic(op.getOperation(), op.getMemref(), slot);
  return mlir::success();
}

static mlir::LogicalResult
verifyContainerLock(mlir::memref::AtomicRMWOp op, ::llvm::StringRef ordering,
                    int64_t value, OrderingRule orderingRule,
                    ::llvm::StringRef what, bool acquire) {
  mlir::FailureOr<ContainerSlot> slot =
      resolveContainerSlot(op, /*lock=*/true, what);
  if (mlir::failed(slot) || mlir::failed(verifyLockMemRef(op, what)) ||
      mlir::failed(verifyKindValue(op, mlir::arith::AtomicRMWKind::assign,
                                   value, what,
                                   value == 0 ? "assign 0" : "assign 1")) ||
      mlir::failed(
          verifyOrdering(op.getOperation(), ordering, orderingRule, what)))
    return mlir::failure();
  if (acquire && mlir::failed(lock::controlFlow(op)))
    return mlir::failure();
  threadsafe::memref::Atomic::set(
      op.getOperation(), ContainerSafetyAttrs::kComponentLock, slot->slot,
      resource::group(op.getMemref()), slot->kind);
  return mlir::success();
}

static mlir::LogicalResult
verifyClassLock(mlir::memref::AtomicRMWOp op, ::llvm::StringRef ordering,
                int64_t value, OrderingRule orderingRule,
                ::llvm::StringRef what, bool acquire) {
  if (mlir::failed(verifyLockMemRef(op, what)) ||
      mlir::failed(memref_index::single(op, 0, what)) ||
      mlir::failed(verifyKindValue(op, mlir::arith::AtomicRMWKind::assign,
                                   value, what,
                                   value == 0 ? "assign 0" : "assign 1")) ||
      mlir::failed(
          verifyOrdering(op.getOperation(), ordering, orderingRule, what)))
    return mlir::failure();
  if (acquire && mlir::failed(lock::controlFlow(op)))
    return mlir::failure();
  resource::sealAtomic(op.getOperation(), op.getMemref(), 0);
  return mlir::success();
}

static mlir::LogicalResult
verifyAsyncCancelRequest(mlir::memref::AtomicRMWOp op,
                         ::llvm::StringRef ordering, const Spec &spec) {
  auto flagIndex = op.getIndices().size() == 1
                       ? constant::index(op.getIndices().front())
                       : std::nullopt;
  if (!flagIndex || *flagIndex != 0)
    return op->emitOpError("async cancel request must target flag slot 0");
  if (mlir::failed(verifyKindValue(op, mlir::arith::AtomicRMWKind::maxu,
                                   spec.value, spec.what, "be maxu 1")) ||
      mlir::failed(verifyOrdering(op.getOperation(), ordering, spec.ordering,
                                  spec.what)))
    return mlir::failure();
  return mlir::success();
}

} // namespace atomic_rmw

namespace generic_refcount {

static bool cmpEq(mlir::Value condition, mlir::Value current,
                  int64_t expected) {
  auto cmp = condition.getDefiningOp<mlir::arith::CmpIOp>();
  if (!cmp || cmp.getPredicate() != mlir::arith::CmpIPredicate::eq)
    return false;
  return (cmp.getLhs() == current &&
          constant::memrefInt(cmp.getRhs(), expected)) ||
         (cmp.getRhs() == current &&
          constant::memrefInt(cmp.getLhs(), expected));
}

static bool cmpSgt(mlir::Value condition, mlir::Value current,
                   int64_t expected) {
  auto cmp = condition.getDefiningOp<mlir::arith::CmpIOp>();
  return cmp && cmp.getPredicate() == mlir::arith::CmpIPredicate::sgt &&
         cmp.getLhs() == current && constant::memrefInt(cmp.getRhs(), expected);
}

static bool addOne(mlir::Value value, mlir::Value current) {
  auto add = value.getDefiningOp<mlir::arith::AddIOp>();
  return add &&
         ((add.getLhs() == current && constant::memrefInt(add.getRhs(), 1)) ||
          (add.getRhs() == current && constant::memrefInt(add.getLhs(), 1)));
}

static bool subOne(mlir::Value value, mlir::Value current) {
  auto sub = value.getDefiningOp<mlir::arith::SubIOp>();
  return sub && sub.getLhs() == current && constant::memrefInt(sub.getRhs(), 1);
}

static mlir::LogicalResult verifyTransition(mlir::memref::GenericAtomicRMWOp op,
                                            bool retain) {
  mlir::Block &block = op.getRegion().front();
  auto yield =
      mlir::dyn_cast<mlir::memref::AtomicYieldOp>(block.getTerminator());
  if (!yield)
    return op->emitOpError("object refcount generic atomic must end with "
                           "memref.atomic_yield");

  mlir::Value current = op.getCurrentValue();
  auto immortalSelect =
      yield.getResult().getDefiningOp<mlir::arith::SelectOp>();
  if (!immortalSelect || immortalSelect.getTrueValue() != current ||
      !cmpEq(immortalSelect.getCondition(), current,
             object_abi::kImmortalRefcount))
    return op->emitOpError()
           << "object " << (retain ? "retain" : "release")
           << " generic atomic must preserve immortal refcounts";

  auto positiveSelect =
      immortalSelect.getFalseValue().getDefiningOp<mlir::arith::SelectOp>();
  if (!positiveSelect || positiveSelect.getFalseValue() != current ||
      !cmpSgt(positiveSelect.getCondition(), current, 0))
    return op->emitOpError()
           << "object " << (retain ? "retain" : "release")
           << " generic atomic must update only positive refcounts";

  bool validUpdate = retain ? addOne(positiveSelect.getTrueValue(), current)
                            : subOne(positiveSelect.getTrueValue(), current);
  if (!validUpdate)
    return op->emitOpError()
           << "object " << (retain ? "retain" : "release")
           << " generic atomic must " << (retain ? "increment" : "decrement")
           << " the current refcount by exactly one";
  return mlir::success();
}

} // namespace generic_refcount

mlir::LogicalResult
verifier::memref::AtomicRMW::verify(mlir::memref::AtomicRMWOp op) {
  auto role = attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicRole);
  if (!role)
    return op->emitOpError("memref atomic is missing ly.atomic.role");

  auto ordering =
      attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicOrdering);
  if (!ordering)
    return op->emitOpError("thread-safety atomic contract is missing ordering");

  const atomic_rmw::Spec *spec = atomic_rmw::lookup(*role);
  if (!spec)
    return op->emitOpError("unsupported ly.atomic.role: ") << *role;

  switch (spec->target) {
  case atomic_rmw::Target::AsyncExceptionLoad:
    return atomic_rmw::verifyAsyncExceptionLoad(op, *ordering);
  case atomic_rmw::Target::ContainerRefcount:
    return atomic_rmw::verifyContainerRefcount(op, *ordering, spec->value,
                                               spec->ordering, spec->what,
                                               spec->retainPremise);
  case atomic_rmw::Target::ObjectRefcount:
    return atomic_rmw::verifyObjectRefcount(
        op, *ordering, object_abi::kRefcountSlot, spec->value, spec->ordering,
        spec->what, spec->retainPremise);
  case atomic_rmw::Target::ClassRefcount:
    return atomic_rmw::verifyObjectRefcount(
        op, *ordering, class_layout::Header::kRefcountSlot, spec->value,
        spec->ordering, spec->what, spec->retainPremise);
  case atomic_rmw::Target::ContainerLock:
    return atomic_rmw::verifyContainerLock(op, *ordering, spec->value,
                                           spec->ordering, spec->what,
                                           spec->acquireControlFlow);
  case atomic_rmw::Target::ClassLock:
    return atomic_rmw::verifyClassLock(op, *ordering, spec->value,
                                       spec->ordering, spec->what,
                                       spec->acquireControlFlow);
  case atomic_rmw::Target::AsyncCancelRequest:
    return atomic_rmw::verifyAsyncCancelRequest(op, *ordering, *spec);
  }

  return op->emitOpError("unsupported ly.atomic.role: ") << *role;
}

mlir::LogicalResult verifier::memref::GenericAtomicRMW::verify(
    mlir::memref::GenericAtomicRMWOp op) {
  auto role = attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicRole);
  if (!role)
    return op->emitOpError("memref generic atomic is missing ly.atomic.role");
  bool asyncStore = *role == ThreadSafetyAttrs::kRoleAsyncExceptionStore;
  bool retain = *role == ThreadSafetyAttrs::kRoleObjectRefcountRetain;
  bool release = *role == ThreadSafetyAttrs::kRoleObjectRefcountRelease;
  if (!asyncStore && !retain && !release)
    return op->emitOpError("unsupported memref generic atomic role: ") << *role;

  if (asyncStore) {
    auto ordering =
        attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicOrdering);
    if (!ordering)
      return op->emitOpError(
          "thread-safety atomic contract is missing ordering");
    if (!ordering::atLeastAcqRel(*ordering))
      return op->emitOpError(
          "async exception cell store must be acq_rel or stronger");
    if (!::py::async_runtime::isExceptionCellType(op.getMemRefType()))
      return op->emitOpError(
          "async exception cell store must target an exception-cell memref");
    if (mlir::failed(memref_index::single(op, 0, "async exception cell store")))
      return mlir::failure();

    mlir::Region &body = op.getRegion();
    if (!body.hasOneBlock())
      return op->emitOpError(
          "async exception cell store must have one body block");
    mlir::Block &block = body.front();
    auto yield =
        mlir::dyn_cast<mlir::memref::AtomicYieldOp>(block.getTerminator());
    if (!yield)
      return op->emitOpError("async exception cell store must end with "
                             "memref.atomic_yield");
    auto select = yield.getResult().getDefiningOp<mlir::arith::SelectOp>();
    if (!select || select.getFalseValue() != op.getCurrentValue())
      return op->emitOpError(
          "async exception cell store must keep the current state on mismatch");
    auto cmp = select.getCondition().getDefiningOp<mlir::arith::CmpIOp>();
    if (!cmp || cmp.getPredicate() != mlir::arith::CmpIPredicate::eq)
      return op->emitOpError(
          "async exception cell store must compare the current state");
    bool reservation = op->hasAttr(AsyncSafetyAttrs::kExceptionCellReservation);
    int64_t expected = reservation ? 0 : 1;
    bool comparesExpected = (cmp.getLhs() == op.getCurrentValue() &&
                             constant::memrefInt(cmp.getRhs(), expected)) ||
                            (cmp.getRhs() == op.getCurrentValue() &&
                             constant::memrefInt(cmp.getLhs(), expected));
    if (!comparesExpected)
      return op->emitOpError(
          "async exception cell store compares the wrong state");
    if (reservation && !constant::memrefInt(select.getTrueValue(), 1))
      return op->emitOpError(
          "async exception cell reservation must store publishing state");
    if (!reservation && constant::memrefInt(select.getTrueValue(), 1))
      return op->emitOpError(
          "async exception cell publish must store a payload state");

    ::py::async_runtime::ExceptionCell::mark(op.getMemref());
    op->setAttr(AsyncSafetyAttrs::kExceptionCellConditionalStore,
                mlir::UnitAttr::get(op.getContext()));
    return mlir::success();
  }

  ::llvm::StringRef action = retain ? "retain" : "release";

  auto ordering =
      attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicOrdering);
  if (!ordering)
    return op->emitOpError("thread-safety atomic contract is missing ordering");
  if (!ordering::atLeastAcqRel(*ordering))
    return op->emitOpError("object ")
           << action << " cmpxchg must be acq_rel or stronger";

  if (!object_header::provenance(op.getMemref()))
    return op->emitOpError("object ")
           << action << " must target a proven object header memref";
  std::string what = (::llvm::Twine("object ") + action + " cmpxchg").str();
  if (mlir::failed(memref_index::single(op, object_abi::kRefcountSlot, what)))
    return mlir::failure();
  if (retain && mlir::failed(verifier::refcount::RetainPremise::verify(
                    op.getOperation())))
    return mlir::failure();

  mlir::Region &body = op.getRegion();
  if (!body.hasOneBlock())
    return op->emitOpError("object ")
           << action << " cmpxchg must have one body block";
  mlir::Block &block = body.front();
  auto yield =
      mlir::dyn_cast<mlir::memref::AtomicYieldOp>(block.getTerminator());
  if (!yield)
    return op->emitOpError("object ")
           << action << " cmpxchg must end with memref.atomic_yield";
  if (yield.getResult().getType() != op.getResult().getType())
    return op->emitOpError("object ")
           << action << " cmpxchg yield type mismatch";
  if (mlir::failed(generic_refcount::verifyTransition(op, retain)))
    return mlir::failure();

  threadsafe::memref::Atomic::set(
      op.getOperation(), ContainerSafetyAttrs::kComponentHeader,
      object_abi::kRefcountSlot, resource::group(op.getMemref()),
      ::llvm::StringRef{});
  mlir::OpBuilder builder(op.getContext());
  op->setAttr(
      ThreadSafetyAttrs::kAtomicProvenance,
      builder.getStringAttr(ThreadSafetyAttrs::kProvenanceMemRefDescriptor));
  return mlir::success();
}

mlir::LogicalResult verifier::memref::Store::verify(mlir::memref::StoreOp op) {
  if (::py::async_runtime::isExceptionCellType(op.getMemRefType())) {
    if (op.getIndices().size() != 1)
      return op->emitOpError(
          "async exception cell store must use one constant slot");
    auto index = constant::index(op.getIndices().front());
    if (!index)
      return op->emitOpError(
          "async exception cell store slot must be constant");

    ::py::async_runtime::ExceptionCell::mark(op.getMemref());

    if (op->hasAttr(AsyncSafetyAttrs::kExceptionCellPayloadStore)) {
      auto intType = mlir::dyn_cast<mlir::IntegerType>(op.getValue().getType());
      if (!intType || intType.getWidth() != 64)
        return op->emitOpError(
            "async exception cell payload store must write an i64 descriptor "
            "word");
      int64_t slots = op.getMemRefType().getShape()[0];
      if (*index <= 0 || *index >= slots)
        return op->emitOpError(
            "async exception cell payload store must target descriptor slots");
      return mlir::success();
    }

    if (*index == 0 && constant::memrefInt(op.getValue(), 0))
      return mlir::success();
    return op->emitOpError("direct async exception cell store must be either "
                           "empty-state initialization or a marked payload "
                           "descriptor store");
  }

  auto refcountSlot = header_slot::refcount(op.getMemref());
  if (!refcountSlot || op.getIndices().size() != 1)
    return mlir::success();
  auto index = constant::index(op.getIndices().front());
  if (!index)
    return op->emitOpError("store into typed container header must use a "
                           "constant slot so refcount writes are auditable");
  if (*index != *refcountSlot)
    return mlir::success();

  mlir::Value value = op.getValue();
  if (constant::memrefInt(value, 0)) {
    if (!memref_value::alloca(op.getMemref()))
      return op->emitOpError("refcount-zero header initialization is only "
                             "allowed for local memref.alloca containers");
    return mlir::success();
  }

  auto initialRefcount = constant::anyInt(value);
  if (initialRefcount && *initialRefcount > 0)
    return refcount_init::verifyManagedStore(op, *initialRefcount);

  return op->emitOpError(
      "direct store to container refcount slot is not a valid initialization");
}

mlir::LogicalResult
verifier::container::HeaderSlot::verify(mlir::Operation *op,
                                        ::llvm::StringRef role) {
  enum class SlotKind { Refcount, Lock };
  struct Spec {
    ::llvm::StringLiteral role;
    ::llvm::StringLiteral component;
    SlotKind kind;
    ::llvm::StringLiteral what;
  };
  static constexpr Spec specs[] = {
      {ThreadSafetyAttrs::kRoleContainerRefcountLoad,
       ContainerSafetyAttrs::kComponentHeader, SlotKind::Refcount,
       "container refcount atomic"},
      {ThreadSafetyAttrs::kRoleContainerRefcountRetain,
       ContainerSafetyAttrs::kComponentHeader, SlotKind::Refcount,
       "container refcount atomic"},
      {ThreadSafetyAttrs::kRoleContainerRefcountRelease,
       ContainerSafetyAttrs::kComponentHeader, SlotKind::Refcount,
       "container refcount atomic"},
      {ThreadSafetyAttrs::kRoleContainerLockAcquire,
       ContainerSafetyAttrs::kComponentLock, SlotKind::Lock,
       "container lock atomic"},
      {ThreadSafetyAttrs::kRoleContainerLockRelease,
       ContainerSafetyAttrs::kComponentLock, SlotKind::Lock,
       "container lock atomic"},
  };
  const Spec *spec = nullptr;
  for (const Spec &candidate : specs)
    if (role == candidate.role) {
      spec = &candidate;
      break;
    }
  if (!spec)
    return mlir::success();

  auto component = attrs::str(op, ThreadSafetyAttrs::kAtomicMemRefComponent);
  auto containerKind = attrs::str(op, ThreadSafetyAttrs::kAtomicContainerKind);
  if (!containerKind)
    return op->emitOpError(
        "container atomic is missing exact container-kind provenance");

  auto slotAttr = op->getAttrOfType<mlir::IntegerAttr>(
      ThreadSafetyAttrs::kAtomicMemRefSlot);
  if (!slotAttr)
    return op->emitOpError(
        "container atomic is missing header slot provenance");

  int64_t slot = slotAttr.getInt();
  if (!component || *component != spec->component)
    return op->emitOpError() << spec->what << " is missing " << spec->component
                             << "-component memref provenance";

  auto expected = spec->kind == SlotKind::Refcount
                      ? header_slot::expectedRefcount(*containerKind)
                      : header_slot::expectedLock(*containerKind);
  if (!expected || *expected != slot)
    return op->emitOpError()
           << spec->what << " targets slot " << slot << " but "
           << *containerKind << " expects slot " << (expected ? *expected : -1);
  return mlir::success();
}

} // namespace py::threadsafe
