#include "Verifier.h"

namespace py::threadsafe {

namespace retain {

static bool lockedBorrow(mlir::Operation *op) {
  if (!retain_op::atomic(op) && !retain_op::runtimeCall(op))
    return false;
  auto premise = attrs::str(op, ThreadSafetyAttrs::kRetainPremise);
  return premise && *premise == ThreadSafetyAttrs::kPremiseLockedBorrow;
}

} // namespace retain

namespace provenance {

static mlir::LLVM::LoadOp classFieldLoad(mlir::Value value) {
  value = pointer::stripCasts(value);
  if (auto gep = value.getDefiningOp<mlir::LLVM::GEPOp>())
    return classFieldLoad(gep.getBase());
  if (auto extract = value.getDefiningOp<mlir::LLVM::ExtractValueOp>())
    return classFieldLoad(extract.getContainer());
  if (auto load = value.getDefiningOp<mlir::LLVM::LoadOp>())
    if (load.getAddr().getDefiningOp<mlir::LLVM::GEPOp>())
      return load;
  return {};
}

static mlir::memref::LoadOp memrefSlotLoad(mlir::Value value) {
  value = pointer::stripCasts(value);
  if (auto load = value.getDefiningOp<mlir::memref::LoadOp>())
    return load;
  return {};
}

static mlir::LLVM::LoadOp llvmSlotLoad(mlir::Value value) {
  value = pointer::stripCasts(value);
  if (auto load = value.getDefiningOp<mlir::LLVM::LoadOp>())
    if (load->hasAttr(ContainerSafetyAttrs::kAccessGroup))
      return load;
  return {};
}

} // namespace provenance

namespace retain {

static mlir::Value pointer(mlir::Operation *retain) {
  if (auto atomic = mlir::dyn_cast<mlir::LLVM::AtomicRMWOp>(retain))
    return atomic.getPtr();
  if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(retain))
    if (call.getNumOperands() >= 1)
      return call.getOperand(0);
  return {};
}

} // namespace retain

namespace lock {

static bool sameMemRef(mlir::Operation *lock, mlir::Operation *retain) {
  mlir::Value lockHeader = atomic::memrefHeader(lock);
  mlir::Value retainHeader = atomic::memrefHeader(retain);
  return lockHeader && retainHeader && lockHeader == retainHeader;
}

static bool sameLLVM(mlir::Operation *lock, mlir::Operation *retain) {
  mlir::Value lockRoot = atomic::llvmRoot(lock);
  mlir::Value retainRoot = atomic::llvmRoot(retain);
  return lockRoot && retainRoot && lockRoot == retainRoot;
}

static bool classFieldRetain(mlir::Operation *lock, mlir::Operation *retain) {
  auto lockRole = attrs::str(lock, ThreadSafetyAttrs::kAtomicRole);
  if (!lockRole || *lockRole != ThreadSafetyAttrs::kRoleClassLockAcquire)
    return false;

  mlir::Value lockRoot = atomic::llvmRoot(lock);
  mlir::Value retainPointer = retain::pointer(retain);
  if (!lockRoot || !retainPointer)
    return false;

  mlir::LLVM::LoadOp fieldLoad = provenance::classFieldLoad(retainPointer);
  if (!fieldLoad)
    return false;

  mlir::Value fieldRoot = pointer::gepRoot(fieldLoad.getAddr());
  return fieldRoot && fieldRoot == lockRoot;
}

static bool containerSlotRetain(mlir::Operation *lock,
                                mlir::Operation *retain) {
  auto lockRole = attrs::str(lock, ThreadSafetyAttrs::kAtomicRole);
  if (!lockRole || *lockRole != ThreadSafetyAttrs::kRoleContainerLockAcquire)
    return false;
  mlir::Value lockHeader = atomic::memrefHeader(lock);
  auto lockGroup = attrs::str(lock, ThreadSafetyAttrs::kAtomicMemRefGroup);
  if (!lockHeader && !lockGroup)
    return false;
  mlir::Value retainPointer = retain::pointer(retain);
  if (!retainPointer)
    return false;
  mlir::memref::LoadOp slotLoad = provenance::memrefSlotLoad(retainPointer);
  if (slotLoad)
    return lockHeader &&
           descriptor::sameResource(lockHeader, slotLoad.getMemref());

  if (auto slotLoad = provenance::llvmSlotLoad(retainPointer)) {
    auto accessGroup =
        attrs::str(slotLoad.getOperation(), ContainerSafetyAttrs::kAccessGroup);
    return lockGroup && accessGroup && *lockGroup == *accessGroup;
  }

  return false;
}

static bool protectsRetain(mlir::Operation *lock, mlir::Operation *retain) {
  return sameMemRef(lock, retain) || sameLLVM(lock, retain) ||
         classFieldRetain(lock, retain) || containerSlotRetain(lock, retain);
}

static bool same(mlir::Operation *lhs, mlir::Operation *rhs) {
  auto lhsGroup = attrs::str(lhs, ThreadSafetyAttrs::kAtomicMemRefGroup);
  auto rhsGroup = attrs::str(rhs, ThreadSafetyAttrs::kAtomicMemRefGroup);
  if (lhsGroup || rhsGroup)
    return lhsGroup && rhsGroup && *lhsGroup == *rhsGroup;

  mlir::Value lhsMemRef = atomic::memrefHeader(lhs);
  mlir::Value rhsMemRef = atomic::memrefHeader(rhs);
  if (lhsMemRef || rhsMemRef)
    return lhsMemRef && rhsMemRef && lhsMemRef == rhsMemRef;

  mlir::Value lhsRoot = atomic::llvmRoot(lhs);
  mlir::Value rhsRoot = atomic::llvmRoot(rhs);
  return lhsRoot && rhsRoot && lhsRoot == rhsRoot;
}

} // namespace lock

mlir::LogicalResult
verifier::container::BorrowRetain::dominance(mlir::Operation *funcLike) {
  mlir::DominanceInfo dominance(funcLike);
  mlir::PostDominanceInfo postDominance(funcLike);
  mlir::SmallVector<mlir::Operation *> acquires;
  mlir::SmallVector<mlir::Operation *> releases;
  mlir::SmallVector<mlir::Operation *> retains;

  funcLike->walk([&](mlir::Operation *op) {
    if (retain::lockedBorrow(op)) {
      retains.push_back(op);
      return;
    }
    auto role = attrs::str(op, ThreadSafetyAttrs::kAtomicRole);
    if (!role)
      return;
    if (role::lockAcquire(*role))
      acquires.push_back(op);
    else if (role::lockRelease(*role))
      releases.push_back(op);
  });

  for (mlir::Operation *retain : retains) {
    mlir::Operation *protectingAcquire = nullptr;
    for (mlir::Operation *acquire : acquires) {
      if (lock::protectsRetain(acquire, retain) &&
          dominance.dominates(acquire, retain)) {
        protectingAcquire = acquire;
        break;
      }
    }
    if (!protectingAcquire)
      return retain->emitOpError(
          "locked-borrow retain is not dominated by a protecting lock acquire");

    for (mlir::Operation *release : releases) {
      if (lock::same(protectingAcquire, release) &&
          dominance.dominates(protectingAcquire, release) &&
          dominance.properlyDominates(release, retain))
        return retain->emitOpError(
            "locked-borrow retain is dominated by a prior protecting lock "
            "release");
    }

    bool hasPostDominatingRelease = false;
    for (mlir::Operation *release : releases) {
      if (!lock::same(protectingAcquire, release))
        continue;
      if (!dominance.dominates(protectingAcquire, release))
        continue;
      if (postDominance.postDominates(release, retain)) {
        hasPostDominatingRelease = true;
        break;
      }
    }
    if (!hasPostDominatingRelease)
      return retain->emitOpError(
          "locked-borrow retain is not followed by a protecting lock release "
          "on every exit path");
  }
  return mlir::success();
}

namespace refcount {

static mlir::memref::AtomicRMWOp loadAtomic(mlir::Value value) {
  auto atomic = value.getDefiningOp<mlir::memref::AtomicRMWOp>();
  if (!atomic)
    return {};
  auto role = attrs::str(atomic.getOperation(), ThreadSafetyAttrs::kAtomicRole);
  if (role && *role == ThreadSafetyAttrs::kRoleContainerRefcountLoad)
    return atomic;
  return {};
}

static bool initStore(mlir::memref::StoreOp store) {
  auto refcountSlot = header_slot::refcount(store.getMemref());
  if (!refcountSlot || store.getIndices().size() != 1)
    return false;
  auto index = constant::index(store.getIndices().front());
  if (!index || *index != *refcountSlot)
    return false;
  return constant::memrefInt(store.getValue(), 0) ||
         constant::memrefInt(store.getValue(), 1);
}

static bool zeroInit(mlir::Value header) {
  auto refcountSlot = header_slot::refcount(header);
  if (!refcountSlot)
    return false;
  for (mlir::Operation *user : header.getUsers()) {
    auto store = mlir::dyn_cast<mlir::memref::StoreOp>(user);
    if (!store || store.getMemref() != header || store.getIndices().size() != 1)
      continue;
    auto index = constant::index(store.getIndices().front());
    if (index && *index == *refcountSlot &&
        constant::memrefInt(store.getValue(), 0))
      return true;
  }
  return false;
}

} // namespace refcount

namespace predicate {

static mlir::Value header(mlir::Value condition) {
  auto cmp = condition.getDefiningOp<mlir::arith::CmpIOp>();
  if (!cmp || cmp.getPredicate() != mlir::arith::CmpIPredicate::ne)
    return {};

  if (constant::memrefInt(cmp.getRhs(), 0)) {
    if (auto atomic = refcount::loadAtomic(cmp.getLhs()))
      return atomic.getMemref();
  }
  if (constant::memrefInt(cmp.getLhs(), 0)) {
    if (auto atomic = refcount::loadAtomic(cmp.getRhs()))
      return atomic.getMemref();
  }
  return {};
}

} // namespace predicate

namespace access {

static mlir::Value target(mlir::Operation *op) {
  if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op))
    return load.getMemref();
  if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op))
    return store.getMemref();
  return {};
}

} // namespace access

namespace nesting {

static bool inside(mlir::Operation *op, mlir::Block *block) {
  for (mlir::Operation *cursor = op; cursor; cursor = cursor->getParentOp())
    if (cursor->getBlock() == block)
      return true;
  return false;
}

} // namespace nesting

namespace fresh {

static bool inside(mlir::Value memref, mlir::Block *block) {
  memref = pointer::stripCasts(memref);
  mlir::Operation *def = memref.getDefiningOp();
  if (!def || !mlir::isa<mlir::memref::AllocaOp, mlir::memref::AllocOp>(def))
    return false;
  return nesting::inside(def, block);
}

} // namespace fresh

namespace local {

static bool zeroAllocaSafe(mlir::Value header) {
  header = pointer::stripCasts(header);
  if (!header.getDefiningOp<mlir::memref::AllocaOp>())
    return false;
  if (!refcount::zeroInit(header))
    return false;
  ::llvm::SmallPtrSet<mlir::Value, 8> seen;
  return local_container::escape(header, seen) == nullptr;
}

} // namespace local

namespace descriptor {

static bool localZeroHeader(mlir::Value target) {
  target = pointer::stripCasts(target);
  auto group = descriptor::group(target);
  if (!group)
    return false;
  mlir::Operation *def = target.getDefiningOp();
  if (!def)
    return false;
  mlir::Operation *scope = nullptr;
  if (auto fn = def->getParentOfType<mlir::func::FuncOp>())
    scope = fn.getOperation();
  if (!scope)
    if (auto fn = def->getParentOfType<mlir::async::FuncOp>())
      scope = fn.getOperation();
  if (!scope)
    if (auto fn = def->getParentOfType<mlir::LLVM::LLVMFuncOp>())
      scope = fn.getOperation();
  if (!scope)
    if (auto module = def->getParentOfType<mlir::ModuleOp>())
      scope = module.getOperation();
  if (!scope)
    return false;

  bool found = false;
  scope->walk([&](mlir::memref::AllocaOp candidate) {
    if (found)
      return;
    auto candidateGroup = descriptor::group(candidate.getResult());
    if (candidateGroup && *candidateGroup == *group &&
        local::zeroAllocaSafe(candidate.getResult()))
      found = true;
  });
  return found;
}

} // namespace descriptor

namespace escape {

static void collect(mlir::Value value,
                    ::llvm::SmallPtrSetImpl<mlir::Value> &seen,
                    mlir::SmallVectorImpl<mlir::Operation *> &escapes) {
  if (!seen.insert(value).second)
    return;
  for (mlir::Operation *user : value.getUsers()) {
    if (local_container::escapeUser(user)) {
      escapes.push_back(user);
      continue;
    }
    if (local_container::use(user, value))
      continue;
    if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(user)) {
      collect(cast.getResult(), seen, escapes);
      continue;
    }
    escapes.push_back(user);
  }
}

} // namespace escape

namespace cfg {

static bool reaches(mlir::Block *from, mlir::Block *to) {
  if (!from || !to)
    return false;
  if (from == to)
    return true;

  mlir::SmallVector<mlir::Block *> worklist;
  ::llvm::SmallPtrSet<mlir::Block *, 16> seen;
  seen.insert(from);
  for (mlir::Block *successor : from->getSuccessors())
    worklist.push_back(successor);

  while (!worklist.empty()) {
    mlir::Block *block = worklist.pop_back_val();
    if (!seen.insert(block).second)
      continue;
    if (block == to)
      return true;
    for (mlir::Block *successor : block->getSuccessors())
      worklist.push_back(successor);
  }
  return false;
}

} // namespace cfg

namespace exec {

static bool before(mlir::Operation *before, mlir::Operation *after) {
  if (!before || !after)
    return false;
  mlir::Block *beforeBlock = before->getBlock();
  mlir::Block *afterBlock = after->getBlock();
  if (beforeBlock == afterBlock)
    return before->isBeforeInBlock(after);
  return cfg::reaches(beforeBlock, afterBlock);
}

} // namespace exec

namespace access {

static bool freshBeforeEscape(mlir::Operation *access, mlir::Value target) {
  target = pointer::stripCasts(target);
  if (!target.getDefiningOp<mlir::memref::AllocaOp>() &&
      !target.getDefiningOp<mlir::memref::AllocOp>())
    return false;

  if (target.getDefiningOp<mlir::memref::AllocaOp>()) {
    if (local::zeroAllocaSafe(target))
      return true;
    if (descriptor::localZeroHeader(target)) {
      ::llvm::SmallPtrSet<mlir::Value, 8> seen;
      if (local_container::escape(target, seen) == nullptr)
        return true;
    }
  }

  mlir::SmallVector<mlir::Operation *> escapes;
  ::llvm::SmallPtrSet<mlir::Value, 8> seen;
  escape::collect(target, seen, escapes);
  for (mlir::Operation *escape : escapes)
    if (exec::before(escape, access))
      return false;
  return true;
}

} // namespace access

static bool protectsAccess(mlir::Operation *acquire, mlir::Operation *access,
                           mlir::Value header,
                           ::llvm::ArrayRef<mlir::Operation *> releases,
                           mlir::DominanceInfo &dominance,
                           mlir::PostDominanceInfo &postDominance) {
  auto acquireRole = attrs::str(acquire, ThreadSafetyAttrs::kAtomicRole);
  if (!acquireRole ||
      *acquireRole != ThreadSafetyAttrs::kRoleContainerLockAcquire)
    return false;
  mlir::Value acquireHeader = atomic::memrefHeader(acquire);
  if (!descriptor::sameResource(header, acquireHeader))
    return false;
  if (!dominance.dominates(acquire, access))
    return false;

  for (mlir::Operation *release : releases) {
    if (!lock::same(acquire, release))
      continue;
    if (dominance.properlyDominates(release, access))
      return false;
  }

  for (mlir::Operation *release : releases) {
    if (!lock::same(acquire, release))
      continue;
    if (!dominance.dominates(acquire, release))
      continue;
    if (postDominance.postDominates(release, access))
      return true;
  }
  return false;
}

mlir::LogicalResult
verifier::container::Access::regions(mlir::Operation *funcLike) {
  mlir::DominanceInfo dominance(funcLike);
  mlir::PostDominanceInfo postDominance(funcLike);
  bool failedAny = false;

  funcLike->walk([&](mlir::scf::IfOp ifOp) {
    mlir::Value header = predicate::header(ifOp.getCondition());
    if (!header)
      return;

    mlir::Block *thenBlock = ifOp.thenBlock();
    mlir::SmallVector<mlir::Operation *> acquires;
    mlir::SmallVector<mlir::Operation *> releases;
    mlir::SmallVector<mlir::Operation *> accesses;

    ifOp.getThenRegion().walk([&](mlir::Operation *nested) {
      auto role = attrs::str(nested, ThreadSafetyAttrs::kAtomicRole);
      if (role && *role == ThreadSafetyAttrs::kRoleContainerLockAcquire) {
        acquires.push_back(nested);
        return;
      }
      if (role && *role == ThreadSafetyAttrs::kRoleContainerLockRelease) {
        releases.push_back(nested);
        return;
      }

      mlir::Value target = access::target(nested);
      if (!target)
        return;
      if (fresh::inside(target, thenBlock))
        return;
      if (descriptor::sameResource(header, target))
        accesses.push_back(nested);
    });

    for (mlir::Operation *accessOp : accesses) {
      bool protectedByLock = false;
      for (mlir::Operation *acquire : acquires) {
        if (protectsAccess(acquire, accessOp, header, releases, dominance,
                           postDominance)) {
          protectedByLock = true;
          break;
        }
      }
      if (!protectedByLock) {
        accessOp->emitOpError(
            "managed container load/store is not protected by the "
            "descriptor's lock acquire/release");
        failedAny = true;
      } else {
        resource::sealAccess(accessOp, header, access::target(accessOp));
      }
    }
  });

  return mlir::failure(failedAny);
}

mlir::LogicalResult
verifier::container::Access::coverage(mlir::Operation *funcLike) {
  bool failedAny = false;
  funcLike->walk([&](mlir::Operation *op) {
    mlir::Value target = access::target(op);
    if (!target)
      return;
    mlir::Value header = descriptor::headerSibling(target);
    if (!header)
      return;
    if (op->hasAttr(ContainerSafetyAttrs::kAccessGroup))
      return;
    if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op))
      if (::py::threadsafe::refcount::initStore(store))
        return;
    if (access::freshBeforeEscape(op, target))
      return;

    op->emitOpError("managed container load/store lacks a sealed access "
                    "contract; shared descriptor accesses must be discovered "
                    "and protected by the container lock verifier");
    failedAny = true;
  });
  return mlir::failure(failedAny);
}

static bool protectsFinalAccess(mlir::Operation *acquire,
                                mlir::Operation *access,
                                ::llvm::ArrayRef<mlir::Operation *> releases,
                                mlir::DominanceInfo &dominance,
                                mlir::PostDominanceInfo &postDominance) {
  auto acquireRole = attrs::str(acquire, ThreadSafetyAttrs::kAtomicRole);
  if (!acquireRole ||
      *acquireRole != ThreadSafetyAttrs::kRoleContainerLockAcquire)
    return false;
  auto acquireGroup =
      attrs::str(acquire, ThreadSafetyAttrs::kAtomicMemRefGroup);
  auto accessGroup = attrs::str(access, ContainerSafetyAttrs::kAccessGroup);
  if (!acquireGroup || !accessGroup || *acquireGroup != *accessGroup)
    return false;
  if (!dominance.dominates(acquire, access))
    return false;

  for (mlir::Operation *release : releases) {
    if (!lock::same(acquire, release))
      continue;
    if (dominance.properlyDominates(release, access))
      return false;
  }

  for (mlir::Operation *release : releases) {
    if (!lock::same(acquire, release))
      continue;
    if (!dominance.dominates(acquire, release))
      continue;
    if (postDominance.postDominates(release, access))
      return true;
  }
  return false;
}

mlir::LogicalResult
verifier::container::Access::final(mlir::Operation *funcLike) {
  mlir::DominanceInfo dominance(funcLike);
  mlir::PostDominanceInfo postDominance(funcLike);
  mlir::SmallVector<mlir::Operation *> acquires;
  mlir::SmallVector<mlir::Operation *> releases;
  mlir::SmallVector<mlir::Operation *> accesses;

  funcLike->walk([&](mlir::Operation *op) {
    auto role = attrs::str(op, ThreadSafetyAttrs::kAtomicRole);
    if (role && *role == ThreadSafetyAttrs::kRoleContainerLockAcquire) {
      acquires.push_back(op);
      return;
    }
    if (role && *role == ThreadSafetyAttrs::kRoleContainerLockRelease) {
      releases.push_back(op);
      return;
    }
    if (op->hasAttr(ContainerSafetyAttrs::kAccessGroup))
      accesses.push_back(op);
  });

  for (mlir::Operation *accessOp : accesses) {
    bool protectedByLock = false;
    for (mlir::Operation *acquire : acquires) {
      if (protectsFinalAccess(acquire, accessOp, releases, dominance,
                              postDominance)) {
        protectedByLock = true;
        break;
      }
    }
    if (!protectedByLock)
      return accessOp->emitOpError(
          "lowered managed container load/store is not protected by the "
          "descriptor's final LLVM lock acquire/release");
  }

  return mlir::success();
}

mlir::LogicalResult
verifier::container::DescriptorAccess::final(mlir::Operation *funcLike) {
  bool failedAny = false;
  funcLike->walk([&](mlir::LLVM::LoadOp load) {
    if (!load->hasAttr(ContainerSafetyAttrs::kAccessComponent))
      return;
    if (load->hasAttr(ContainerSafetyAttrs::kAccessGroup))
      return;
    load->emitOpError("lowered managed container load preserved an access "
                      "component without its access_group");
    failedAny = true;
  });
  funcLike->walk([&](mlir::LLVM::StoreOp store) {
    if (!store->hasAttr(ContainerSafetyAttrs::kAccessComponent))
      return;
    if (store->hasAttr(ContainerSafetyAttrs::kAccessGroup))
      return;
    store->emitOpError("lowered managed container store preserved an access "
                       "component without its access_group");
    failedAny = true;
  });
  return mlir::failure(failedAny);
}

} // namespace py::threadsafe
