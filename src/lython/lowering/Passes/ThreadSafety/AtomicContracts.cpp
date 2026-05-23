#include "Verifier.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

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

static mlir::LLVM::LLVMFuncOp lookupLLVMFunc(mlir::Operation *context,
                                             ::llvm::StringRef name) {
  if (!context)
    return {};
  auto module = context->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return {};
  return module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name);
}

namespace class_helper {

static bool hasKind(mlir::Operation *context, ::llvm::StringRef name,
                    ::llvm::StringRef expectedKind) {
  auto fn = lookupLLVMFunc(context, name);
  if (!fn)
    return false;
  auto helperKind =
      attrs::str(fn.getOperation(), ClassSafetyAttrs::kHelperKind);
  auto helperClass =
      attrs::str(fn.getOperation(), ClassSafetyAttrs::kHelperClass);
  return helperKind && helperClass && *helperKind == expectedKind;
}

} // namespace class_helper

bool retain_op::runtimeCall(mlir::Operation *op) {
  auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(op);
  if (!call)
    return false;
  auto callee = call.getCallee();
  return callee &&
         (*callee == RuntimeSymbols::kIncRef ||
          class_helper::hasKind(op, *callee, ClassSafetyAttrs::kKindIncref));
}

bool retain_op::atomic(mlir::Operation *op) {
  auto role = attrs::str(op, ThreadSafetyAttrs::kAtomicRole);
  return role && (*role == ThreadSafetyAttrs::kRoleContainerRefcountRetain ||
                  *role == ThreadSafetyAttrs::kRoleClassRefcountRetain);
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
  if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>())
    if (cast->getNumOperands() == 1)
      return borrow(cast.getOperand(0), seen);

  if (auto gep = value.getDefiningOp<mlir::LLVM::GEPOp>())
    return borrow(gep.getBase(), seen);
  if (auto extract = value.getDefiningOp<mlir::LLVM::ExtractValueOp>())
    return borrow(extract.getContainer(), seen);

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
  if (auto atomic = mlir::dyn_cast<mlir::LLVM::AtomicRMWOp>(op))
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

  ::llvm::StringRef name = fn.getName();
  if (*role == ThreadSafetyAttrs::kRoleClassRefcountRetain &&
      *helperKind == ClassSafetyAttrs::kKindIncref)
    return mlir::success();

  if (*role != ThreadSafetyAttrs::kRoleContainerRefcountRetain)
    return op->emitOpError("class-helper owned-token proof is not valid for "
                           "this retain role");

  if (*helperKind == ClassSafetyAttrs::kKindGetField) {
    if (branchDominates(op, fn, /*argIndex=*/1, /*trueEdge=*/false))
      return mlir::success();
    return op->emitOpError("getfield helper owned-token retain is not "
                           "dominated by the local-retain branch");
  }

  if (*helperKind == ClassSafetyAttrs::kKindSetField) {
    if (branchDominates(op, fn, /*argIndex=*/2, /*trueEdge=*/true))
      return mlir::success();
    return op->emitOpError("setfield helper owned-token retain is not "
                           "dominated by the retain-new-value branch");
  }

  return op->emitOpError("class-helper owned-token proof appears in an "
                         "unsupported helper: ")
         << name;
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
    if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(user)) {
      for (mlir::Value result : cast.getResults())
        collectEscapes(result, seen, escapes);
      continue;
    }
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
  for (mlir::Operation *resultUser : op->getResult(0).getUsers()) {
    auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(resultUser);
    if (!cmp ||
        !cmp::memrefZero(cmp, op->getResult(0), mlir::arith::CmpIPredicate::ne))
      continue;
    for (mlir::Operation *cmpUser : cmp->getResult(0).getUsers()) {
      auto condition = mlir::dyn_cast<mlir::scf::ConditionOp>(cmpUser);
      if (!condition || condition.getCondition() != cmp->getResult(0))
        continue;
      if (condition->getParentOfType<mlir::scf::WhileOp>())
        return mlir::success();
    }
  }
  return op->emitOpError("container lock acquire result must be checked by "
                         "a spin-loop condition that repeats while the "
                         "previous lock value is non-zero");
}

} // namespace lock

mlir::LogicalResult
verifier::memref::AtomicRMW::verify(mlir::memref::AtomicRMWOp op) {
  auto role = attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicRole);
  if (!role)
    return op->emitOpError("memref atomic is missing ly.atomic.role");

  auto ordering =
      attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicOrdering);
  if (!ordering)
    return op->emitOpError("thread-safety atomic contract is missing ordering");

  mlir::arith::AtomicRMWKind kind = op.getKind();
  mlir::Value value = op.getValue();

  if (*role == ThreadSafetyAttrs::kRoleContainerRefcountLoad) {
    auto refcountSlot = header_slot::refcount(op.getMemref());
    if (!refcountSlot)
      return op->emitOpError("container refcount load must target a known "
                             "typed-container header");
    auto containerKind = descriptor::Kind::get(op.getMemref());
    auto expectedSlot = containerKind
                            ? header_slot::expectedRefcount(*containerKind)
                            : std::optional<int64_t>{};
    if (!containerKind || !expectedSlot || *expectedSlot != *refcountSlot)
      return op->emitOpError(
          "container refcount load lacks exact typed header kind provenance");
    if (mlir::failed(
            memref_index::single(op, *refcountSlot, "container refcount load")))
      return mlir::failure();
    if (kind != mlir::arith::AtomicRMWKind::addi ||
        !constant::memrefInt(value, 0))
      return op->emitOpError("container refcount load must be addi 0");
    if (!ordering::atLeastAcquire(*ordering))
      return op->emitOpError(
          "container refcount load must be acquire or stronger");
    resource::sealAtomic(op.getOperation(), op.getMemref(), *refcountSlot);
    return mlir::success();
  }

  if (*role == ThreadSafetyAttrs::kRoleContainerRefcountRetain) {
    auto refcountSlot = header_slot::refcount(op.getMemref());
    if (!refcountSlot)
      return op->emitOpError("container retain must target a known "
                             "typed-container header");
    auto containerKind = descriptor::Kind::get(op.getMemref());
    auto expectedSlot = containerKind
                            ? header_slot::expectedRefcount(*containerKind)
                            : std::optional<int64_t>{};
    if (!containerKind || !expectedSlot || *expectedSlot != *refcountSlot)
      return op->emitOpError(
          "container retain lacks exact typed header kind provenance");
    if (mlir::failed(
            memref_index::single(op, *refcountSlot, "container retain")))
      return mlir::failure();
    if (memref_value::alloca(op.getMemref()))
      return op->emitOpError("container retain must not mutate a local "
                             "refcount-zero alloca header");
    if (mlir::failed(
            verifier::refcount::RetainPremise::verify(op.getOperation())))
      return mlir::failure();
    if (kind != mlir::arith::AtomicRMWKind::addi ||
        !constant::memrefInt(value, 1))
      return op->emitOpError("container retain must be addi +1");
    if (!ordering::refcountInc(*ordering))
      return op->emitOpError("container retain must be monotonic or stronger");
    resource::sealAtomic(op.getOperation(), op.getMemref(), *refcountSlot);
    return mlir::success();
  }

  if (*role == ThreadSafetyAttrs::kRoleContainerRefcountRelease) {
    auto refcountSlot = header_slot::refcount(op.getMemref());
    if (!refcountSlot)
      return op->emitOpError("container release must target a known "
                             "typed-container header");
    auto containerKind = descriptor::Kind::get(op.getMemref());
    auto expectedSlot = containerKind
                            ? header_slot::expectedRefcount(*containerKind)
                            : std::optional<int64_t>{};
    if (!containerKind || !expectedSlot || *expectedSlot != *refcountSlot)
      return op->emitOpError(
          "container release lacks exact typed header kind provenance");
    if (mlir::failed(
            memref_index::single(op, *refcountSlot, "container release")))
      return mlir::failure();
    if (memref_value::alloca(op.getMemref()))
      return op->emitOpError("container release must not mutate a local "
                             "refcount-zero alloca header");
    if (kind != mlir::arith::AtomicRMWKind::addi ||
        !constant::memrefInt(value, -1))
      return op->emitOpError("container release must be addi -1");
    if (!ordering::atLeastAcqRel(*ordering))
      return op->emitOpError("container release must be acq_rel or stronger");
    resource::sealAtomic(op.getOperation(), op.getMemref(), *refcountSlot);
    return mlir::success();
  }

  if (*role == ThreadSafetyAttrs::kRoleContainerLockAcquire) {
    auto lockSlot = header_slot::lock(op.getMemref());
    if (!lockSlot)
      return op->emitOpError("container lock acquire must target a known "
                             "mutable typed-container header");
    auto containerKind = descriptor::Kind::get(op.getMemref());
    auto expectedSlot = containerKind
                            ? header_slot::expectedLock(*containerKind)
                            : std::optional<int64_t>{};
    if (!containerKind || !expectedSlot || *expectedSlot != *lockSlot)
      return op->emitOpError(
          "container lock acquire lacks exact mutable header kind provenance");
    if (mlir::failed(
            memref_index::single(op, *lockSlot, "container lock acquire")))
      return mlir::failure();
    if (kind != mlir::arith::AtomicRMWKind::assign ||
        !constant::memrefInt(value, 1))
      return op->emitOpError("container lock acquire must assign 1");
    if (!ordering::atLeastAcquire(*ordering))
      return op->emitOpError(
          "container lock acquire must be acquire or stronger");
    if (mlir::failed(lock::controlFlow(op)))
      return mlir::failure();
    resource::sealAtomic(op.getOperation(), op.getMemref(), *lockSlot);
    return mlir::success();
  }

  if (*role == ThreadSafetyAttrs::kRoleContainerLockRelease) {
    auto lockSlot = header_slot::lock(op.getMemref());
    if (!lockSlot)
      return op->emitOpError("container lock release must target a known "
                             "mutable typed-container header");
    auto containerKind = descriptor::Kind::get(op.getMemref());
    auto expectedSlot = containerKind
                            ? header_slot::expectedLock(*containerKind)
                            : std::optional<int64_t>{};
    if (!containerKind || !expectedSlot || *expectedSlot != *lockSlot)
      return op->emitOpError(
          "container lock release lacks exact mutable header kind provenance");
    if (mlir::failed(
            memref_index::single(op, *lockSlot, "container lock release")))
      return mlir::failure();
    if (kind != mlir::arith::AtomicRMWKind::assign ||
        !constant::memrefInt(value, 0))
      return op->emitOpError("container lock release must assign 0");
    if (!ordering::atLeastRelease(*ordering))
      return op->emitOpError(
          "container lock release must be release or stronger");
    resource::sealAtomic(op.getOperation(), op.getMemref(), *lockSlot);
    return mlir::success();
  }

  if (*role == ThreadSafetyAttrs::kRoleAsyncCancelRequest) {
    auto flagIndex = op.getIndices().size() == 1
                         ? constant::index(op.getIndices().front())
                         : std::nullopt;
    if (!flagIndex || *flagIndex != 0)
      return op->emitOpError("async cancel request must target flag slot 0");
    if (kind != mlir::arith::AtomicRMWKind::maxu ||
        !constant::memrefInt(value, 1))
      return op->emitOpError("async cancel request must be maxu 1");
    if (!ordering::atLeastAcqRel(*ordering))
      return op->emitOpError(
          "async cancel request must be acq_rel or stronger");
    return mlir::success();
  }

  return op->emitOpError("unsupported ly.atomic.role: ") << *role;
}

mlir::LogicalResult verifier::memref::Store::verify(mlir::memref::StoreOp op) {
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
  auto component = attrs::str(op, ThreadSafetyAttrs::kAtomicMemRefComponent);
  if (!component || *component != ContainerSafetyAttrs::kComponentHeader)
    return op->emitOpError("container atomic is missing header-component "
                           "memref provenance");

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
  if (role == ThreadSafetyAttrs::kRoleContainerRefcountLoad ||
      role == ThreadSafetyAttrs::kRoleContainerRefcountRetain ||
      role == ThreadSafetyAttrs::kRoleContainerRefcountRelease) {
    auto expected = header_slot::expectedRefcount(*containerKind);
    if (!expected || *expected != slot)
      return op->emitOpError("container refcount atomic targets header slot ")
             << slot << " but " << *containerKind << " expects refcount slot "
             << (expected ? *expected : -1);
    return mlir::success();
  }

  if (role == ThreadSafetyAttrs::kRoleContainerLockAcquire ||
      role == ThreadSafetyAttrs::kRoleContainerLockRelease) {
    auto expected = header_slot::expectedLock(*containerKind);
    if (!expected || *expected != slot)
      return op->emitOpError("container lock atomic targets header slot ")
             << slot << " but " << *containerKind << " expects lock slot "
             << (expected ? *expected : -1);
    return mlir::success();
  }

  return mlir::success();
}

} // namespace py::threadsafe
