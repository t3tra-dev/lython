#include "Verifier.h"

namespace py::threadsafe {

bool local_container::escapeUser(mlir::Operation *op) {
  return mlir::isa<mlir::func::ReturnOp, mlir::LLVM::ReturnOp,
                   mlir::async::ReturnOp, mlir::async::RuntimeStoreOp,
                   mlir::func::CallOp, mlir::LLVM::CallOp>(op);
}

bool local_container::use(mlir::Operation *op, mlir::Value value) {
  if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op))
    return load.getMemref() == value;
  if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op))
    return store.getMemref() == value;
  if (auto atomic = mlir::dyn_cast<mlir::memref::AtomicRMWOp>(op))
    return atomic.getMemref() == value;
  if (auto dealloc = mlir::dyn_cast<mlir::memref::DeallocOp>(op))
    return dealloc.getMemref() == value;
  return false;
}

mlir::Operation *
local_container::escape(mlir::Value value,
                        ::llvm::SmallPtrSetImpl<mlir::Value> &seen) {
  if (!seen.insert(value).second)
    return nullptr;

  for (mlir::Operation *user : value.getUsers()) {
    if (local_container::escapeUser(user))
      return user;

    if (local_container::use(user, value))
      continue;

    if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(user)) {
      if (mlir::Operation *escape =
              local_container::escape(cast.getResult(), seen))
        return escape;
      continue;
    }

    return user;
  }
  return nullptr;
}

namespace release {

static mlir::memref::AtomicRMWOp memrefAtomic(mlir::Value value) {
  auto atomic = value.getDefiningOp<mlir::memref::AtomicRMWOp>();
  if (!atomic)
    return {};
  auto role = attrs::str(atomic.getOperation(), ThreadSafetyAttrs::kAtomicRole);
  if (role && *role == ThreadSafetyAttrs::kRoleContainerRefcountRelease)
    return atomic;
  return {};
}

static mlir::memref::AtomicRMWOp toZero(mlir::Value condition,
                                        mlir::Value deallocated) {
  auto cmp = condition.getDefiningOp<mlir::arith::CmpIOp>();
  if (!cmp || cmp.getPredicate() != mlir::arith::CmpIPredicate::eq)
    return {};

  auto matches = [&](mlir::Value lhs, mlir::Value rhs) {
    auto release = memrefAtomic(lhs);
    if (!release || !constant::memrefInt(rhs, 1))
      return mlir::memref::AtomicRMWOp{};
    if (!descriptor::sameResource(release.getMemref(), deallocated))
      return mlir::memref::AtomicRMWOp{};
    return release;
  };
  if (auto release = matches(cmp.getLhs(), cmp.getRhs()))
    return release;
  return matches(cmp.getRhs(), cmp.getLhs());
}

static void sealDealloc(mlir::memref::DeallocOp dealloc,
                        mlir::memref::AtomicRMWOp release) {
  if (!dealloc || !release)
    return;
  auto group =
      attrs::str(release.getOperation(), ThreadSafetyAttrs::kAtomicMemRefGroup);
  if (!group)
    return;
  mlir::OpBuilder builder(dealloc.getContext());
  dealloc->setAttr(ContainerSafetyAttrs::kDeallocGroup,
                   builder.getStringAttr(*group));
  if (auto component = descriptor::component(dealloc.getMemref()))
    dealloc->setAttr(ContainerSafetyAttrs::kDeallocComponent,
                     builder.getStringAttr(*component));
}

} // namespace release

mlir::LogicalResult
verifier::memref::Dealloc::verify(mlir::memref::DeallocOp dealloc) {
  mlir::Value memref = dealloc.getMemref();
  if (memref_value::alloca(memref))
    return mlir::success();

  for (mlir::Operation *parent = dealloc->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(parent);
    if (!ifOp)
      continue;
    if (auto release = release::toZero(ifOp.getCondition(), memref)) {
      release::sealDealloc(dealloc, release);
      return mlir::success();
    }
  }

  if (!descriptor::value(memref))
    return mlir::success();

  return dealloc->emitOpError("managed container dealloc is not guarded by "
                              "refcount release returning 1");
}

namespace release {

static mlir::LLVM::AtomicRMWOp llvmAtomic(mlir::Value value) {
  auto atomic = value.getDefiningOp<mlir::LLVM::AtomicRMWOp>();
  if (!atomic)
    return {};
  auto role = attrs::str(atomic.getOperation(), ThreadSafetyAttrs::kAtomicRole);
  if (role && *role == ThreadSafetyAttrs::kRoleContainerRefcountRelease)
    return atomic;
  return {};
}

} // namespace release

bool control::llvmReleaseToZero(mlir::Value condition,
                                ::llvm::StringRef group) {
  auto cmp = condition.getDefiningOp<mlir::LLVM::ICmpOp>();
  if (!cmp || cmp.getPredicate() != mlir::LLVM::ICmpPredicate::eq)
    return false;

  auto matches = [&](mlir::Value lhs, mlir::Value rhs) {
    auto release = release::llvmAtomic(lhs);
    if (!release || !constant::llvmInt(rhs, 1))
      return false;
    auto releaseGroup = attrs::str(release.getOperation(),
                                   ThreadSafetyAttrs::kAtomicMemRefGroup);
    return releaseGroup && *releaseGroup == group;
  };
  return matches(cmp.getLhs(), cmp.getRhs()) ||
         matches(cmp.getRhs(), cmp.getLhs());
}

mlir::Value control::condition(mlir::Operation *terminator) {
  if (auto branch = mlir::dyn_cast<mlir::cf::CondBranchOp>(terminator))
    return branch.getCondition();
  if (auto branch = mlir::dyn_cast<mlir::LLVM::CondBrOp>(terminator))
    return branch.getCondition();
  return {};
}

mlir::LogicalResult verifier::llvm::FreeCall::verify(mlir::LLVM::CallOp call) {
  auto group =
      attrs::str(call.getOperation(), ContainerSafetyAttrs::kDeallocGroup);
  if (!group)
    return mlir::success();

  auto fn = call->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  if (!fn)
    return call->emitOpError("managed container free is outside an LLVM "
                             "function");

  if (call.getNumOperands() < 1 ||
      !provenance::descriptorAllocated(call.getOperand(0)))
    return call->emitOpError("managed container free does not target a lowered "
                             "memref descriptor allocated pointer");

  mlir::DominanceInfo dominance(fn.getOperation());
  bool guarded = false;
  fn.walk([&](mlir::Operation *terminator) {
    if (guarded || terminator->getNumSuccessors() < 2)
      return;
    mlir::Value condition = control::condition(terminator);
    if (!condition || !control::llvmReleaseToZero(condition, *group))
      return;
    mlir::Block *freeSuccessor = terminator->getSuccessor(0);
    guarded = ::py::threadsafe::dominance::block(
        freeSuccessor, call.getOperation(), dominance);
  });

  if (guarded)
    return mlir::success();
  return call->emitOpError("managed container free is not guarded by "
                           "refcount release returning 1");
}

} // namespace py::threadsafe
