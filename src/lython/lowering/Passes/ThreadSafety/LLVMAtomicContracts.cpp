#include "Verifier.h"

namespace py::threadsafe {

namespace atomic_role {

static mlir::LogicalResult verify(mlir::LLVM::AtomicRMWOp op,
                                  ::llvm::StringRef role) {
  if (role::containerAtomic(role)) {
    auto provenance =
        attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicProvenance);
    if (!provenance ||
        *provenance != ThreadSafetyAttrs::kProvenanceMemRefDescriptor)
      return op->emitOpError("container LLVM atomic must declare "
                             "memref-descriptor provenance");
    if (!attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicMemRefGroup))
      return op->emitOpError(
          "container LLVM atomic is missing memref resource group provenance");
    if (!provenance::descriptorData(op.getPtr()))
      return op->emitOpError("container LLVM atomic does not target lowered "
                             "memref descriptor data");
    if (mlir::failed(
            verifier::container::HeaderSlot::verify(op.getOperation(), role)))
      return mlir::failure();
  }

  if (role == ThreadSafetyAttrs::kRoleAsyncCancelRequest) {
    if (!provenance::asyncCancelFlag(op.getOperation(), op.getPtr()))
      return op->emitOpError(
          "async cancel request lacks cancel-flag provenance");
    mlir::LLVM::AtomicBinOp binOp = op.getBinOp();
    mlir::LLVM::AtomicOrdering ordering = op.getOrdering();
    mlir::Value value = op.getVal();
    if (mlir::failed(verifier::llvm::Ordering::verify(op, ordering)))
      return mlir::failure();
    if (binOp != mlir::LLVM::AtomicBinOp::umax || !constant::llvmInt(value, 1))
      return op->emitOpError("async cancel request must be atomic umax 1");
    if (!ordering::atLeastAcqRel(ordering))
      return op->emitOpError("async cancel request must be acq_rel or seq_cst");
    return mlir::success();
  }

  if (auto provenance =
          attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicProvenance)) {
    if (*provenance != ThreadSafetyAttrs::kProvenanceMemRefDescriptor)
      return op->emitOpError("unsupported LLVM atomic provenance: ")
             << *provenance;
    if (!provenance::descriptorData(op.getPtr()))
      return op->emitOpError("LLVM atomic declares memref-descriptor "
                             "provenance but does not target lowered memref "
                             "descriptor data");
  } else if (role::containerAtomic(role)) {
    return op->emitOpError("container LLVM atomic is missing "
                           "memref-descriptor provenance");
  } else if (!provenance::gep(op.getPtr())) {
    return op->emitOpError(
        "LLVM atomic with Lython role must target an address "
        "derived from a GEP provenance");
  }

  mlir::LLVM::AtomicBinOp binOp = op.getBinOp();
  mlir::LLVM::AtomicOrdering ordering = op.getOrdering();
  mlir::Value value = op.getVal();
  if (mlir::failed(verifier::llvm::Ordering::verify(op, ordering)))
    return mlir::failure();

  if (role::retainRefcount(role)) {
    if (mlir::failed(
            verifier::refcount::RetainPremise::verify(op.getOperation())))
      return mlir::failure();
    if (binOp != mlir::LLVM::AtomicBinOp::add || !constant::llvmInt(value, 1))
      return op->emitOpError("shared retain must be atomic add +1");
    if (!ordering::refcountInc(ordering))
      return op->emitOpError("shared retain must be monotonic or stronger");
    return mlir::success();
  }

  if (role::releaseRefcount(role)) {
    if (binOp != mlir::LLVM::AtomicBinOp::add || !constant::llvmInt(value, -1))
      return op->emitOpError("shared release must be atomic add -1");
    if (!ordering::atLeastAcqRel(ordering))
      return op->emitOpError("shared release must be acq_rel or seq_cst");
    return mlir::success();
  }

  if (role == ThreadSafetyAttrs::kRoleContainerRefcountLoad) {
    if (binOp != mlir::LLVM::AtomicBinOp::add || !constant::llvmInt(value, 0))
      return op->emitOpError("shared refcount load must be atomic add 0");
    if (!ordering::atLeastAcquire(ordering))
      return op->emitOpError(
          "shared refcount load must be acquire or stronger");
    return mlir::success();
  }

  if (role == ThreadSafetyAttrs::kRoleClassLockAcquire ||
      role == ThreadSafetyAttrs::kRoleContainerLockAcquire) {
    if (binOp != mlir::LLVM::AtomicBinOp::xchg || !constant::llvmInt(value, 1))
      return op->emitOpError("lock acquire must be atomic xchg 1");
    if (!ordering::atLeastAcquire(ordering))
      return op->emitOpError("lock acquire must be acquire or stronger");
    if (mlir::failed(verifier::llvm::LockAcquire::controlFlow(op)))
      return mlir::failure();
    return mlir::success();
  }

  if (role == ThreadSafetyAttrs::kRoleClassLockRelease ||
      role == ThreadSafetyAttrs::kRoleContainerLockRelease) {
    if (binOp != mlir::LLVM::AtomicBinOp::xchg || !constant::llvmInt(value, 0))
      return op->emitOpError("lock release must be atomic xchg 0");
    if (!ordering::atLeastRelease(ordering))
      return op->emitOpError("lock release must be release or stronger");
    return mlir::success();
  }

  return op->emitOpError("unsupported LLVM atomic role: ") << role;
}

} // namespace atomic_role

mlir::LogicalResult
verifier::llvm::AtomicRMW::verify(mlir::LLVM::AtomicRMWOp op) {
  if (auto role = attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicRole))
    return atomic_role::verify(op, *role);

  return op->emitOpError("LLVM atomic RMW is missing ly.atomic.role");
}

mlir::LogicalResult
verifier::llvm::RetainCall::verify(mlir::LLVM::CallOp call) {
  return verifier::refcount::RetainPremise::verify(call.getOperation());
}

mlir::LogicalResult verifier::llvm::Store::verify(mlir::LLVM::StoreOp op) {
  if (auto init = op->getAttrOfType<mlir::IntegerAttr>(
          ContainerSafetyAttrs::kRefcountInit)) {
    auto state =
        attrs::str(op.getOperation(), ContainerSafetyAttrs::kRefcountState);
    if (!state || *state != ContainerSafetyAttrs::kStateManaged)
      return op->emitOpError(
          "container refcount initialization must declare managed state");
    if (!provenance::gep(op.getAddr()))
      return op->emitOpError("container refcount initialization must target a "
                             "GEP-derived header slot");
    if (init.getInt() <= 0)
      return op->emitOpError(
          "managed container refcount initialization must be positive");
    if (!constant::llvmInt(op.getValue(), init.getInt()))
      return op->emitOpError(
          "container refcount initialization value does not match contract");
  }

  mlir::LLVM::AtomicOrdering ordering = op.getOrdering();
  if (ordering == mlir::LLVM::AtomicOrdering::not_atomic)
    return mlir::success();

  if (!provenance::gep(op.getAddr()))
    return op->emitOpError("atomic store must target a GEP-derived address");

  if (auto role =
          attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicRole)) {
    if (*role != ThreadSafetyAttrs::kRoleClassLockRelease &&
        *role != ThreadSafetyAttrs::kRoleContainerLockRelease)
      return op->emitOpError("unsupported LLVM atomic store role: ") << *role;
    if (*role == ThreadSafetyAttrs::kRoleContainerLockRelease) {
      auto provenance =
          attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicProvenance);
      if (!provenance ||
          *provenance != ThreadSafetyAttrs::kProvenanceMemRefDescriptor)
        return op->emitOpError("container lock release store must declare "
                               "memref-descriptor provenance");
      if (!attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicMemRefGroup))
        return op->emitOpError("container lock release store is missing "
                               "memref resource group provenance");
      if (mlir::failed(verifier::container::HeaderSlot::verify(
              op.getOperation(), *role)))
        return mlir::failure();
      if (!provenance::descriptorData(op.getAddr()))
        return op->emitOpError("container lock release store does not target "
                               "lowered memref descriptor data");
    }
    if (!constant::llvmInt(op.getValue(), 0))
      return op->emitOpError("lock release store must write 0");
    if (!ordering::atLeastRelease(ordering))
      return op->emitOpError("lock release store must be release or stronger");
    return mlir::success();
  }

  return op->emitOpError("atomic store is missing ly.atomic.role");
}

mlir::LogicalResult verifier::llvm::Load::verify(mlir::LLVM::LoadOp op) {
  mlir::LLVM::AtomicOrdering ordering = op.getOrdering();
  auto role = attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicRole);
  if (ordering == mlir::LLVM::AtomicOrdering::not_atomic) {
    if (role)
      return op->emitOpError("non-atomic load carries Lython atomic role");
    return mlir::success();
  }

  if (!role)
    return op->emitOpError("atomic load is missing ly.atomic.role");
  if (*role == ThreadSafetyAttrs::kRoleAsyncExceptionLoad) {
    if (!provenance::asyncExceptionCell(op.getAddr()))
      return op->emitOpError(
          "async exception cell load lacks exception-cell provenance");
    if (!ordering::atLeastAcquire(ordering))
      return op->emitOpError("async exception cell load must be acquire or "
                             "stronger");
    return mlir::success();
  }
  if (*role == ThreadSafetyAttrs::kRoleAsyncCancelLoad) {
    if (!provenance::asyncCancelFlag(op.getOperation(), op.getAddr()))
      return op->emitOpError(
          "async cancel flag load lacks cancel-flag provenance");
    if (!ordering::atLeastAcquire(ordering))
      return op->emitOpError("async cancel flag load must be acquire or "
                             "stronger");
    return mlir::success();
  }
  return op->emitOpError("unsupported LLVM atomic load role: ") << *role;
}

mlir::LogicalResult
verifier::llvm::CmpXchg::verify(mlir::LLVM::AtomicCmpXchgOp op) {
  auto role = attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicRole);
  if (!role)
    return op->emitOpError("atomic cmpxchg is missing ly.atomic.role");
  if (*role != ThreadSafetyAttrs::kRoleAsyncExceptionStore)
    return op->emitOpError("unsupported LLVM atomic cmpxchg role: ") << *role;
  if (!provenance::asyncExceptionCell(op.getPtr()))
    return op->emitOpError(
        "async exception cell cmpxchg lacks exception-cell provenance");
  if (!constant::llvmNullPtr(op.getCmp()))
    return op->emitOpError(
        "async exception cell store must compare against null");
  if (!mlir::isa<mlir::LLVM::LLVMPointerType>(op.getVal().getType()))
    return op->emitOpError(
        "async exception cell payload must be an LLVM pointer");
  if (!ordering::atLeastAcqRel(op.getSuccessOrdering()))
    return op->emitOpError("async exception cell cmpxchg success ordering must "
                           "be acq_rel or stronger");
  if (!ordering::atLeastAcquire(op.getFailureOrdering()))
    return op->emitOpError("async exception cell cmpxchg failure ordering must "
                           "be acquire or stronger");
  return mlir::success();
}

mlir::LogicalResult verifier::async_runtime::RefcountCall::verify(
    mlir::Operation *op, ::llvm::StringRef callee, mlir::ValueRange operands) {
  if (!py::async_runtime::Callee::refcount(callee))
    return mlir::success();
  if (operands.size() != 2)
    return op->emitOpError("MLIR async runtime refcount call must have "
                           "handle and count operands");
  if (!mlir::isa<mlir::LLVM::LLVMPointerType>(operands[0].getType()))
    return op->emitOpError("MLIR async runtime refcount handle must be an "
                           "LLVM pointer");
  auto count = constant::anyInt(operands[1]);
  if (!count || *count <= 0)
    return op->emitOpError("MLIR async runtime refcount count must be a "
                           "positive integer constant");
  return mlir::success();
}

} // namespace py::threadsafe
