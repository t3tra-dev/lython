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

namespace {

namespace host_dealloc {

static bool shape(mlir::LLVM::CallOp call) {
  return call.getNumResults() == 0 && call.getNumOperands() == 1 &&
         mlir::isa<mlir::LLVM::LLVMPointerType>(call.getOperand(0).getType());
}

static mlir::LogicalResult requireShape(mlir::LLVM::CallOp call,
                                        llvm::StringRef contractName) {
  if (shape(call))
    return mlir::success();
  return call->emitOpError()
         << contractName
         << " contract must be attached to a void single-pointer host dealloc "
            "boundary";
}

static mlir::LogicalResult
requireDescriptorAllocated(mlir::LLVM::CallOp call,
                           llvm::StringRef contractName) {
  if (provenance::descriptorAllocated(call.getOperand(0)))
    return mlir::success();
  return call->emitOpError()
         << contractName
         << " contract does not target a lowered memref descriptor allocated "
            "pointer";
}

} // namespace host_dealloc

static bool objectReleaseToZero(mlir::Value condition) {
  condition = pointer::stripCasts(condition);
  if (!condition)
    return false;
  mlir::Operation *def = condition.getDefiningOp();
  if (!def)
    return false;
  if (def->hasAttr(OwnershipContractAttrs::kObjectReleaseToZero))
    return true;
  llvm::StringRef name = def->getName().getStringRef();
  if ((name == "llvm.sext" || name == "llvm.zext" || name == "llvm.trunc") &&
      def->getNumOperands() == 1)
    return objectReleaseToZero(def->getOperand(0));
  return false;
}

static bool guardedByObjectReleaseToZero(mlir::LLVM::LLVMFuncOp fn,
                                         mlir::Operation *dealloc) {
  mlir::DominanceInfo dominance(fn.getOperation());
  bool guarded = false;
  fn.walk([&](mlir::Operation *terminator) {
    if (guarded || terminator->getNumSuccessors() < 2)
      return;
    mlir::Value condition = control::condition(terminator);
    if (!objectReleaseToZero(condition))
      return;
    mlir::Block *deallocSuccessor = terminator->getSuccessor(0);
    guarded = ::py::threadsafe::dominance::block(deallocSuccessor, dealloc,
                                                 dominance);
  });
  return guarded;
}

} // namespace

mlir::Value control::condition(mlir::Operation *terminator) {
  if (auto branch = mlir::dyn_cast<mlir::cf::CondBranchOp>(terminator))
    return branch.getCondition();
  if (auto branch = mlir::dyn_cast<mlir::LLVM::CondBrOp>(terminator))
    return branch.getCondition();
  return {};
}

mlir::LogicalResult verifier::llvm::FreeCall::verify(mlir::LLVM::CallOp call) {
  if (call->hasAttr(AsyncSafetyAttrs::kExceptionCellFree)) {
    if (mlir::failed(
            host_dealloc::requireShape(call, "async exception cell free")))
      return mlir::failure();
    if (!provenance::asyncExceptionCellAllocated(call.getOperand(0)))
      return call->emitOpError(
          "async exception cell free does not target a lowered memref "
          "descriptor allocated pointer");
    return mlir::success();
  }

  if (call->hasAttr(ClassSafetyAttrs::kDeallocPart)) {
    if (mlir::failed(host_dealloc::requireShape(call, "class dealloc")))
      return mlir::failure();
    return host_dealloc::requireDescriptorAllocated(call, "class dealloc");
  }

  if (call->hasAttr(OwnershipContractAttrs::kObjectDeallocPart)) {
    if (mlir::failed(host_dealloc::requireShape(call, "object dealloc")))
      return mlir::failure();
    if (mlir::failed(
            host_dealloc::requireDescriptorAllocated(call, "object dealloc")))
      return mlir::failure();
    auto fn = call->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    if (!fn || !guardedByObjectReleaseToZero(fn, call.getOperation()))
      return call->emitOpError(
          "object dealloc contract is not guarded by object refcount release "
          "returning zero");
    return mlir::success();
  }

  auto group =
      attrs::str(call.getOperation(), ContainerSafetyAttrs::kDeallocGroup);
  if (!group)
    return mlir::success();

  auto fn = call->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  if (!fn)
    return call->emitOpError("managed container free is outside an LLVM "
                             "function");

  if (mlir::failed(host_dealloc::requireShape(call, "managed container")))
    return mlir::failure();
  if (mlir::failed(
          host_dealloc::requireDescriptorAllocated(call, "managed container")))
    return mlir::failure();

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
