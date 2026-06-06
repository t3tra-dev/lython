#include "Verifier.h"

#include "Common/ClassLayout.h"
#include "Common/Object.h"

namespace py::threadsafe {

namespace atomic_role {

enum class OrderingRule { Acquire, Release, AcqRel, RefcountInc };

enum class Target {
  AsyncCancelFlag,
  AsyncExceptionCell,
  Container,
  ObjectRefcount,
  ClassRefcount,
  ClassLock,
};

struct Spec {
  ::llvm::StringLiteral role;
  Target target;
  mlir::LLVM::AtomicBinOp binOp;
  int64_t value;
  OrderingRule ordering;
  ::llvm::StringLiteral what;
  bool retainPremise = false;
  bool acquireControlFlow = false;
};

static constexpr Spec kSpecs[] = {
    {ThreadSafetyAttrs::kRoleAsyncCancelRequest, Target::AsyncCancelFlag,
     mlir::LLVM::AtomicBinOp::umax, 1, OrderingRule::AcqRel,
     "async cancel request"},
    {ThreadSafetyAttrs::kRoleAsyncExceptionLoad, Target::AsyncExceptionCell,
     mlir::LLVM::AtomicBinOp::add, 0, OrderingRule::Acquire,
     "async exception cell load"},
    {ThreadSafetyAttrs::kRoleContainerRefcountLoad, Target::Container,
     mlir::LLVM::AtomicBinOp::add, 0, OrderingRule::Acquire,
     "container refcount load"},
    {ThreadSafetyAttrs::kRoleContainerRefcountRetain, Target::Container,
     mlir::LLVM::AtomicBinOp::add, 1, OrderingRule::RefcountInc,
     "container retain", /*retainPremise=*/true},
    {ThreadSafetyAttrs::kRoleContainerRefcountRelease, Target::Container,
     mlir::LLVM::AtomicBinOp::add, -1, OrderingRule::AcqRel,
     "container release"},
    {ThreadSafetyAttrs::kRoleObjectRefcountLoad, Target::ObjectRefcount,
     mlir::LLVM::AtomicBinOp::add, 0, OrderingRule::Acquire,
     "object refcount load"},
    {ThreadSafetyAttrs::kRoleObjectRefcountRetain, Target::ObjectRefcount,
     mlir::LLVM::AtomicBinOp::add, 1, OrderingRule::RefcountInc,
     "object retain", /*retainPremise=*/true},
    {ThreadSafetyAttrs::kRoleObjectRefcountRelease, Target::ObjectRefcount,
     mlir::LLVM::AtomicBinOp::add, -1, OrderingRule::AcqRel, "object release"},
    {ThreadSafetyAttrs::kRoleClassRefcountLoad, Target::ClassRefcount,
     mlir::LLVM::AtomicBinOp::add, 0, OrderingRule::Acquire,
     "class refcount load"},
    {ThreadSafetyAttrs::kRoleClassRefcountRetain, Target::ClassRefcount,
     mlir::LLVM::AtomicBinOp::add, 1, OrderingRule::RefcountInc, "class retain",
     /*retainPremise=*/true},
    {ThreadSafetyAttrs::kRoleClassRefcountRelease, Target::ClassRefcount,
     mlir::LLVM::AtomicBinOp::add, -1, OrderingRule::AcqRel, "class release"},
    {ThreadSafetyAttrs::kRoleContainerLockAcquire, Target::Container,
     mlir::LLVM::AtomicBinOp::xchg, 1, OrderingRule::Acquire,
     "container lock acquire", /*retainPremise=*/false,
     /*acquireControlFlow=*/true},
    {ThreadSafetyAttrs::kRoleContainerLockRelease, Target::Container,
     mlir::LLVM::AtomicBinOp::xchg, 0, OrderingRule::Release,
     "container lock release"},
    {ThreadSafetyAttrs::kRoleClassLockAcquire, Target::ClassLock,
     mlir::LLVM::AtomicBinOp::xchg, 1, OrderingRule::Acquire,
     "class lock acquire", /*retainPremise=*/false,
     /*acquireControlFlow=*/true},
    {ThreadSafetyAttrs::kRoleClassLockRelease, Target::ClassLock,
     mlir::LLVM::AtomicBinOp::xchg, 0, OrderingRule::Release,
     "class lock release"},
};

static const Spec *lookup(::llvm::StringRef role) {
  for (const Spec &spec : kSpecs)
    if (role == spec.role)
      return &spec;
  return nullptr;
}

static bool accepts(OrderingRule rule, mlir::LLVM::AtomicOrdering ordering) {
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

static mlir::LogicalResult verifyOrdering(mlir::LLVM::AtomicRMWOp op,
                                          const Spec &spec) {
  if (accepts(spec.ordering, op.getOrdering()))
    return mlir::success();
  return op->emitOpError() << spec.what << " must be "
                           << description(spec.ordering);
}

static mlir::LogicalResult verifyKindValue(mlir::LLVM::AtomicRMWOp op,
                                           const Spec &spec) {
  if (op.getBinOp() == spec.binOp && constant::llvmInt(op.getVal(), spec.value))
    return mlir::success();
  return op->emitOpError() << spec.what << " has invalid atomic operation";
}

static mlir::LogicalResult requireMemRefDescriptor(mlir::LLVM::AtomicRMWOp op,
                                                   ::llvm::StringRef what) {
  auto provenance =
      attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicProvenance);
  if (!provenance ||
      *provenance != ThreadSafetyAttrs::kProvenanceMemRefDescriptor)
    return op->emitOpError()
           << what << " must declare memref-descriptor provenance";
  if (!provenance::descriptorData(op.getPtr()))
    return op->emitOpError()
           << what << " does not target lowered memref descriptor data";
  return mlir::success();
}

static mlir::LogicalResult requirePointerProvenance(mlir::LLVM::AtomicRMWOp op,
                                                    ::llvm::StringRef what) {
  if (auto provenance =
          attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicProvenance)) {
    if (*provenance != ThreadSafetyAttrs::kProvenanceMemRefDescriptor)
      return op->emitOpError("unsupported LLVM atomic provenance: ")
             << *provenance;
    if (provenance::descriptorData(op.getPtr()))
      return mlir::success();
    return op->emitOpError()
           << what << " declares memref-descriptor provenance but does not "
           << "target lowered memref descriptor data";
  }
  if (provenance::gep(op.getPtr()))
    return mlir::success();
  return op->emitOpError()
         << what << " must target an address derived from GEP provenance";
}

static mlir::LogicalResult verifyContainer(mlir::LLVM::AtomicRMWOp op,
                                           ::llvm::StringRef role,
                                           ::llvm::StringRef what) {
  if (mlir::failed(requireMemRefDescriptor(op, what)))
    return mlir::failure();
  if (!attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicMemRefGroup))
    return op->emitOpError()
           << what << " is missing memref resource group provenance";
  return verifier::container::HeaderSlot::verify(op.getOperation(), role);
}

static mlir::LogicalResult verifyObjectRefcount(mlir::LLVM::AtomicRMWOp op,
                                                ::llvm::StringRef what) {
  if (mlir::failed(requireMemRefDescriptor(op, what)))
    return mlir::failure();
  auto slot =
      attrs::i64(op.getOperation(), ThreadSafetyAttrs::kAtomicMemRefSlot);
  if (!slot || *slot != ::py::object_abi::kRefcountSlot)
    return op->emitOpError()
           << what << " must target object header refcount slot";
  return mlir::success();
}

static mlir::LogicalResult verifyClassRefcount(mlir::LLVM::AtomicRMWOp op,
                                               ::llvm::StringRef what) {
  if (mlir::failed(requireMemRefDescriptor(op, what)))
    return mlir::failure();
  auto component =
      attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicMemRefComponent);
  if (!component || *component != ContainerSafetyAttrs::kComponentHeader)
    return op->emitOpError()
           << what << " must target the class header component";
  auto slot =
      attrs::i64(op.getOperation(), ThreadSafetyAttrs::kAtomicMemRefSlot);
  if (!slot || *slot != class_layout::Header::kRefcountSlot)
    return op->emitOpError()
           << what << " must target class header refcount slot";
  return mlir::success();
}

static mlir::LogicalResult verifyProvenance(mlir::LLVM::AtomicRMWOp op,
                                            const Spec &spec) {
  switch (spec.target) {
  case Target::AsyncCancelFlag:
    if (provenance::asyncCancelFlag(op.getOperation(), op.getPtr()))
      return mlir::success();
    return op->emitOpError() << spec.what << " lacks cancel-flag provenance";
  case Target::AsyncExceptionCell:
    if (provenance::asyncExceptionCell(op.getPtr()))
      return mlir::success();
    return op->emitOpError() << spec.what << " lacks exception-cell provenance";
  case Target::Container:
    return verifyContainer(op, spec.role, spec.what);
  case Target::ObjectRefcount:
    return verifyObjectRefcount(op, spec.what);
  case Target::ClassRefcount:
    return verifyClassRefcount(op, spec.what);
  case Target::ClassLock:
    return requirePointerProvenance(op, spec.what);
  }
  return op->emitOpError("unsupported LLVM atomic target");
}

static mlir::LogicalResult verify(mlir::LLVM::AtomicRMWOp op,
                                  ::llvm::StringRef role) {
  const Spec *spec = lookup(role);
  if (!spec)
    return op->emitOpError("unsupported LLVM atomic role: ") << role;

  if (mlir::failed(verifier::llvm::Ordering::verify(op, op.getOrdering())) ||
      mlir::failed(verifyProvenance(op, *spec)) ||
      mlir::failed(verifyKindValue(op, *spec)) ||
      mlir::failed(verifyOrdering(op, *spec)))
    return mlir::failure();

  if (spec->retainPremise &&
      mlir::failed(
          verifier::refcount::RetainPremise::verify(op.getOperation())))
    return mlir::failure();

  if (spec->acquireControlFlow &&
      mlir::failed(verifier::llvm::LockAcquire::controlFlow(op)))
    return mlir::failure();

  return mlir::success();
}

} // namespace atomic_role

namespace cmpxchg {

enum class Target {
  ObjectRefcount,
  AsyncExceptionCell,
};

struct Spec {
  ::llvm::StringLiteral role;
  Target target;
  ::llvm::StringLiteral action;
  bool retainPremise = false;
};

static constexpr Spec kSpecs[] = {
    {ThreadSafetyAttrs::kRoleObjectRefcountRetain, Target::ObjectRefcount,
     "retain", /*retainPremise=*/true},
    {ThreadSafetyAttrs::kRoleObjectRefcountRelease, Target::ObjectRefcount,
     "release"},
    {ThreadSafetyAttrs::kRoleAsyncExceptionStore, Target::AsyncExceptionCell,
     "async exception cell store"},
};

static const Spec *lookup(::llvm::StringRef role) {
  for (const Spec &spec : kSpecs)
    if (role == spec.role)
      return &spec;
  return nullptr;
}

static mlir::LogicalResult verifyObjectRefcount(mlir::LLVM::AtomicCmpXchgOp op,
                                                const Spec &spec) {
  auto provenance =
      attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicProvenance);
  if (!provenance ||
      *provenance != ThreadSafetyAttrs::kProvenanceMemRefDescriptor)
    return op->emitOpError("object ")
           << spec.action
           << " cmpxchg must declare memref-descriptor provenance";

  auto slot =
      attrs::i64(op.getOperation(), ThreadSafetyAttrs::kAtomicMemRefSlot);
  if (!slot || *slot != ::py::object_abi::kRefcountSlot)
    return op->emitOpError("object ")
           << spec.action << " cmpxchg must target object header refcount slot";

  if (!provenance::gep(op.getPtr()))
    return op->emitOpError("object ")
           << spec.action << " cmpxchg must target a GEP-derived address";

  if (!provenance::descriptorData(op.getPtr()))
    return op->emitOpError("object ")
           << spec.action
           << " cmpxchg does not target lowered memref descriptor data";

  if (spec.retainPremise &&
      mlir::failed(
          verifier::refcount::RetainPremise::verify(op.getOperation())))
    return mlir::failure();

  auto cmpType = mlir::dyn_cast<mlir::IntegerType>(op.getCmp().getType());
  auto valueType = mlir::dyn_cast<mlir::IntegerType>(op.getVal().getType());
  if (!cmpType || !valueType || cmpType.getWidth() != 64 ||
      valueType.getWidth() != 64)
    return op->emitOpError("object ")
           << spec.action
           << " cmpxchg must compare and store i64 refcount values";

  if (!ordering::atLeastAcqRel(op.getSuccessOrdering()))
    return op->emitOpError("object ")
           << spec.action
           << " cmpxchg success ordering must be acq_rel or stronger";

  if (!ordering::atLeastAcquire(op.getFailureOrdering()))
    return op->emitOpError("object ")
           << spec.action
           << " cmpxchg failure ordering must be acquire or stronger";

  return mlir::success();
}

static mlir::LogicalResult
verifyAsyncExceptionCell(mlir::LLVM::AtomicCmpXchgOp op) {
  if (!provenance::asyncExceptionCell(op.getPtr()))
    return op->emitOpError(
        "async exception cell cmpxchg lacks exception-cell provenance");
  bool conditionalStore =
      op->hasAttr(AsyncSafetyAttrs::kExceptionCellConditionalStore);
  bool comparesEmpty =
      constant::llvmNullPtr(op.getCmp()) || constant::llvmInt(op.getCmp(), 0);
  bool comparesPublishing = constant::llvmInt(op.getCmp(), 1);
  if (!conditionalStore && !comparesEmpty && !comparesPublishing)
    return op->emitOpError(
        "async exception cell store must compare against empty or publishing");
  auto valueType = op.getVal().getType();
  auto intType = mlir::dyn_cast<mlir::IntegerType>(valueType);
  if (!mlir::isa<mlir::LLVM::LLVMPointerType>(valueType) &&
      !(intType && intType.getWidth() == 64))
    return op->emitOpError(
        "async exception cell payload must be a lowered descriptor payload "
        "word");
  if (!conditionalStore &&
      op->hasAttr(AsyncSafetyAttrs::kExceptionCellReservation) &&
      (!comparesEmpty || !constant::llvmInt(op.getVal(), 1)))
    return op->emitOpError(
        "async exception cell reservation must compare empty and store "
        "publishing");
  if (!conditionalStore &&
      !op->hasAttr(AsyncSafetyAttrs::kExceptionCellReservation) &&
      comparesPublishing && constant::llvmInt(op.getVal(), 1))
    return op->emitOpError(
        "async exception cell publish must replace publishing with payload");
  if (!ordering::atLeastAcqRel(op.getSuccessOrdering()))
    return op->emitOpError("async exception cell cmpxchg success ordering must "
                           "be acq_rel or stronger");
  if (!ordering::atLeastAcquire(op.getFailureOrdering()))
    return op->emitOpError("async exception cell cmpxchg failure ordering must "
                           "be acquire or stronger");
  return mlir::success();
}

static mlir::LogicalResult verify(mlir::LLVM::AtomicCmpXchgOp op,
                                  ::llvm::StringRef role) {
  const Spec *spec = lookup(role);
  if (!spec)
    return op->emitOpError("unsupported LLVM atomic cmpxchg role: ") << role;
  switch (spec->target) {
  case Target::ObjectRefcount:
    return verifyObjectRefcount(op, *spec);
  case Target::AsyncExceptionCell:
    return verifyAsyncExceptionCell(op);
  }
  return op->emitOpError("unsupported LLVM atomic cmpxchg target");
}

} // namespace cmpxchg

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

namespace store_role {

enum class Target {
  ObjectRefcountInit,
  ObjectKindInit,
  ObjectPayloadInit,
  ContainerLockRelease,
  ClassLockRelease,
};

struct Spec {
  ::llvm::StringLiteral role;
  Target target;
  ::llvm::StringLiteral what;
};

static constexpr Spec kSpecs[] = {
    {ThreadSafetyAttrs::kRoleObjectRefcountInit, Target::ObjectRefcountInit,
     "object refcount initialization"},
    {ThreadSafetyAttrs::kRoleObjectKindInit, Target::ObjectKindInit,
     "object kind initialization"},
    {ThreadSafetyAttrs::kRoleObjectPayloadInit, Target::ObjectPayloadInit,
     "object payload initialization"},
    {ThreadSafetyAttrs::kRoleContainerLockRelease, Target::ContainerLockRelease,
     "container lock release store"},
    {ThreadSafetyAttrs::kRoleClassLockRelease, Target::ClassLockRelease,
     "class lock release store"},
};

static const Spec *lookup(::llvm::StringRef role) {
  for (const Spec &spec : kSpecs)
    if (role == spec.role)
      return &spec;
  return nullptr;
}

static mlir::LogicalResult requireRelease(mlir::LLVM::StoreOp op,
                                          const Spec &spec) {
  if (ordering::atLeastRelease(op.getOrdering()))
    return mlir::success();
  return op->emitOpError() << spec.what << " must be release or stronger";
}

static mlir::LogicalResult verifyObjectInit(mlir::LLVM::StoreOp op,
                                            const Spec &spec) {
  if (mlir::failed(requireRelease(op, spec)))
    return mlir::failure();

  switch (spec.target) {
  case Target::ObjectRefcountInit:
    if (mlir::isa<mlir::IntegerType>(op.getValue().getType()))
      return mlir::success();
    return op->emitOpError("object refcount initialization must store an "
                           "integer");
  case Target::ObjectKindInit:
    if (mlir::isa<mlir::LLVM::LLVMPointerType>(op.getValue().getType()))
      return mlir::success();
    return op->emitOpError("object kind initialization must store an LLVM "
                           "pointer");
  case Target::ObjectPayloadInit:
    return mlir::success();
  case Target::ContainerLockRelease:
  case Target::ClassLockRelease:
    break;
  }
  return op->emitOpError("unsupported object initialization store target");
}

static mlir::LogicalResult verifyContainerLockRelease(mlir::LLVM::StoreOp op,
                                                      const Spec &spec) {
  auto provenance =
      attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicProvenance);
  if (!provenance ||
      *provenance != ThreadSafetyAttrs::kProvenanceMemRefDescriptor)
    return op->emitOpError("container lock release store must declare "
                           "memref-descriptor provenance");
  if (!attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicMemRefGroup))
    return op->emitOpError("container lock release store is missing memref "
                           "resource group provenance");
  if (mlir::failed(verifier::container::HeaderSlot::verify(op.getOperation(),
                                                           spec.role)))
    return mlir::failure();
  if (!provenance::descriptorData(op.getAddr()))
    return op->emitOpError("container lock release store does not target "
                           "lowered memref descriptor data");
  return mlir::success();
}

static mlir::LogicalResult verifyLockRelease(mlir::LLVM::StoreOp op,
                                             const Spec &spec) {
  if (spec.target == Target::ContainerLockRelease &&
      mlir::failed(verifyContainerLockRelease(op, spec)))
    return mlir::failure();
  if (!constant::llvmInt(op.getValue(), 0))
    return op->emitOpError("lock release store must write 0");
  return requireRelease(op, spec);
}

static mlir::LogicalResult verify(mlir::LLVM::StoreOp op,
                                  ::llvm::StringRef role) {
  const Spec *spec = lookup(role);
  if (!spec)
    return op->emitOpError("unsupported LLVM atomic store role: ") << role;

  switch (spec->target) {
  case Target::ObjectRefcountInit:
  case Target::ObjectKindInit:
  case Target::ObjectPayloadInit:
    return verifyObjectInit(op, *spec);
  case Target::ContainerLockRelease:
  case Target::ClassLockRelease:
    return verifyLockRelease(op, *spec);
  }
  return op->emitOpError("unsupported LLVM atomic store target");
}

} // namespace store_role

mlir::LogicalResult verifier::llvm::Store::verify(mlir::LLVM::StoreOp op) {
  if (op->hasAttr(AsyncSafetyAttrs::kExceptionCellPayloadStore)) {
    auto intType = mlir::dyn_cast<mlir::IntegerType>(op.getValue().getType());
    if (!intType || intType.getWidth() != 64)
      return op->emitOpError(
          "async exception cell payload store must write an i64 descriptor "
          "word");
    if (!provenance::asyncExceptionCell(op.getAddr()))
      return op->emitOpError(
          "async exception cell payload store lacks exception-cell "
          "provenance");
    return mlir::success();
  }

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
    return store_role::verify(op, *role);
  }

  return op->emitOpError("atomic store is missing ly.atomic.role");
}

namespace load_role {

enum class Target {
  AsyncExceptionCell,
  AsyncCancelFlag,
  ObjectRefcount,
  ObjectPayload,
};

struct Spec {
  ::llvm::StringLiteral role;
  Target target;
  ::llvm::StringLiteral what;
};

static constexpr Spec kSpecs[] = {
    {ThreadSafetyAttrs::kRoleAsyncExceptionLoad, Target::AsyncExceptionCell,
     "async exception cell load"},
    {ThreadSafetyAttrs::kRoleAsyncCancelLoad, Target::AsyncCancelFlag,
     "async cancel flag load"},
    {ThreadSafetyAttrs::kRoleObjectRefcountLoad, Target::ObjectRefcount,
     "object refcount load"},
    {ThreadSafetyAttrs::kRoleObjectPayloadLoad, Target::ObjectPayload,
     "object payload load"},
};

static const Spec *lookup(::llvm::StringRef role) {
  for (const Spec &spec : kSpecs)
    if (role == spec.role)
      return &spec;
  return nullptr;
}

static mlir::LogicalResult requireAcquire(mlir::LLVM::LoadOp op,
                                          const Spec &spec) {
  if (ordering::atLeastAcquire(op.getOrdering()))
    return mlir::success();
  return op->emitOpError() << spec.what << " must be acquire or stronger";
}

static mlir::LogicalResult requireDescriptor(mlir::LLVM::LoadOp op,
                                             const Spec &spec) {
  auto provenance =
      attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicProvenance);
  if (!provenance ||
      *provenance != ThreadSafetyAttrs::kProvenanceMemRefDescriptor)
    return op->emitOpError()
           << spec.what << " must declare memref-descriptor provenance";
  if (!provenance::descriptorData(op.getAddr()))
    return op->emitOpError()
           << spec.what << " does not target lowered memref descriptor data";
  return mlir::success();
}

static mlir::LogicalResult verifyProvenance(mlir::LLVM::LoadOp op,
                                            const Spec &spec) {
  switch (spec.target) {
  case Target::AsyncExceptionCell:
    if (provenance::asyncExceptionCell(op.getAddr()))
      return mlir::success();
    return op->emitOpError() << spec.what << " lacks exception-cell provenance";
  case Target::AsyncCancelFlag:
    if (provenance::asyncCancelFlag(op.getOperation(), op.getAddr()))
      return mlir::success();
    return op->emitOpError() << spec.what << " lacks cancel-flag provenance";
  case Target::ObjectRefcount: {
    if (mlir::failed(requireDescriptor(op, spec)))
      return mlir::failure();
    auto slot =
        attrs::i64(op.getOperation(), ThreadSafetyAttrs::kAtomicMemRefSlot);
    if (!slot || *slot != ::py::object_abi::kRefcountSlot)
      return op->emitOpError()
             << spec.what << " must target object header refcount slot";
    if (provenance::gep(op.getAddr()))
      return mlir::success();
    return op->emitOpError()
           << spec.what << " must target a GEP-derived address";
  }
  case Target::ObjectPayload:
    if (provenance::gep(op.getAddr()))
      return mlir::success();
    return op->emitOpError()
           << spec.what << " must target a GEP-derived address";
  }
  return op->emitOpError("unsupported LLVM atomic load target");
}

static mlir::LogicalResult verify(mlir::LLVM::LoadOp op,
                                  ::llvm::StringRef role) {
  const Spec *spec = lookup(role);
  if (!spec)
    return op->emitOpError("unsupported LLVM atomic load role: ") << role;
  if (mlir::failed(verifyProvenance(op, *spec)) ||
      mlir::failed(requireAcquire(op, *spec)))
    return mlir::failure();
  return mlir::success();
}

} // namespace load_role

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
  return load_role::verify(op, *role);
}

mlir::LogicalResult
verifier::llvm::CmpXchg::verify(mlir::LLVM::AtomicCmpXchgOp op) {
  auto role = attrs::str(op.getOperation(), ThreadSafetyAttrs::kAtomicRole);
  if (!role)
    return op->emitOpError("atomic cmpxchg is missing ly.atomic.role");
  return cmpxchg::verify(op, *role);
}

mlir::LogicalResult
verifier::async_runtime::RefcountCall::verify(mlir::Operation *op,
                                              mlir::ValueRange operands) {
  auto delta = attrs::i64(op, AsyncSafetyAttrs::kRuntimeRefcountDelta);
  if (!delta)
    return mlir::success();
  if (*delta == 0)
    return op->emitOpError("MLIR async runtime refcount delta must be nonzero");
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
  if (*delta != 1 && *delta != -1)
    return op->emitOpError("MLIR async runtime refcount delta must describe "
                           "one retain or one release per count unit");
  return mlir::success();
}

} // namespace py::threadsafe
