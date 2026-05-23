#include "Verifier.h"

namespace py::threadsafe {

namespace scope {

static mlir::Operation *local(mlir::Operation *op) {
  for (mlir::Operation *scope = op; scope; scope = scope->getParentOp())
    if (mlir::isa<mlir::func::FuncOp, mlir::async::FuncOp,
                  mlir::LLVM::LLVMFuncOp, mlir::ModuleOp>(scope))
      return scope;
  return op;
}

} // namespace scope

mlir::LogicalResult
verifier::memref::Alloca::verify(mlir::memref::AllocaOp alloca) {
  auto refcountSlot = header_slot::refcount(alloca.getResult());
  if (!refcountSlot)
    return mlir::success();

  bool hasZeroRefcountInit = false;
  for (mlir::Operation *user : alloca.getResult().getUsers()) {
    auto store = mlir::dyn_cast<mlir::memref::StoreOp>(user);
    if (!store || store.getMemref() != alloca.getResult() ||
        store.getIndices().size() != 1)
      continue;
    auto index = constant::index(store.getIndices().front());
    if (index && *index == *refcountSlot &&
        constant::memrefInt(store.getValue(), 0)) {
      hasZeroRefcountInit = true;
      break;
    }
  }
  if (!hasZeroRefcountInit)
    return mlir::success();

  ::llvm::SmallPtrSet<mlir::Value, 8> seen;
  if (mlir::Operation *escape =
          local_container::escape(alloca.getResult(), seen))
    return escape->emitOpError(
        "refcount-zero local container header escapes its allocation scope");

  std::optional<::llvm::StringRef> group =
      descriptor::group(alloca.getResult());
  if (!group)
    return mlir::success();

  mlir::Operation *verifierScope = scope::local(alloca.getOperation());
  mlir::SmallVector<mlir::memref::AllocaOp> components;
  verifierScope->walk([&](mlir::memref::AllocaOp candidate) {
    auto candidateGroup = descriptor::group(candidate.getResult());
    if (candidateGroup && *candidateGroup == *group)
      components.push_back(candidate);
  });

  for (mlir::memref::AllocaOp component : components) {
    if (component == alloca)
      continue;
    ::llvm::SmallPtrSet<mlir::Value, 8> componentSeen;
    if (mlir::Operation *escape =
            local_container::escape(component.getResult(), componentSeen))
      return escape->emitOpError(
          "refcount-zero local container descriptor component escapes its "
          "allocation scope");
  }
  return mlir::success();
}

bool provenance::asyncExceptionCell(mlir::Value value) {
  while (true) {
    if (auto bitcast = value.getDefiningOp<mlir::LLVM::BitcastOp>()) {
      value = bitcast.getArg();
      continue;
    }
    break;
  }

  if (mlir::isa<mlir::BlockArgument>(value))
    return function_arg::hasAttr(value, AsyncSafetyAttrs::kExceptionCell);
  if (mlir::Operation *def = value.getDefiningOp())
    return def->hasAttr(AsyncSafetyAttrs::kExceptionCell);
  return false;
}

bool provenance::asyncCancelFlag(mlir::Value value) {
  while (true) {
    if (auto bitcast = value.getDefiningOp<mlir::LLVM::BitcastOp>()) {
      value = bitcast.getArg();
      continue;
    }
    if (auto extract = value.getDefiningOp<mlir::LLVM::ExtractValueOp>()) {
      value = extract.getContainer();
      continue;
    }
    if (auto gep = value.getDefiningOp<mlir::LLVM::GEPOp>()) {
      value = gep.getBase();
      continue;
    }
    break;
  }

  if (mlir::isa<mlir::BlockArgument>(value))
    return function_arg::hasAttr(value, AsyncSafetyAttrs::kCancelFlag);
  if (value.getDefiningOp<mlir::LLVM::AllocaOp>())
    return true;
  if (mlir::Operation *def = value.getDefiningOp())
    return def->hasAttr(AsyncSafetyAttrs::kCancelFlag);
  return false;
}

bool provenance::asyncCancelFlag(mlir::Operation *op, mlir::Value value) {
  if (op && op->hasAttr(AsyncSafetyAttrs::kCancelFlag))
    return true;
  return provenance::asyncCancelFlag(value);
}

namespace exception_cell {

namespace users {

static void collect(mlir::Value value,
                    ::llvm::SmallPtrSetImpl<mlir::Value> &seen,
                    mlir::SmallVectorImpl<mlir::Operation *> &users) {
  value = pointer::stripCasts(value);
  if (!seen.insert(value).second)
    return;

  for (mlir::Operation *user : value.getUsers()) {
    if (auto bitcast = mlir::dyn_cast<mlir::LLVM::BitcastOp>(user)) {
      collect(bitcast.getResult(), seen, users);
      continue;
    }
    users.push_back(user);
  }
}

} // namespace users

namespace init {

static bool store(mlir::Value cell, mlir::Operation *user) {
  auto store = mlir::dyn_cast<mlir::LLVM::StoreOp>(user);
  if (!store)
    return false;
  return pointer::stripCasts(store.getAddr()) == pointer::stripCasts(cell) &&
         constant::llvmNullPtr(store.getValue()) &&
         store.getOrdering() == mlir::LLVM::AtomicOrdering::not_atomic;
}

static mlir::LogicalResult verify(mlir::Value cell,
                                  mlir::DominanceInfo &dominance,
                                  mlir::Operation *owner) {
  mlir::SmallVector<mlir::Operation *> users;
  ::llvm::SmallPtrSet<mlir::Value, 8> seen;
  users::collect(cell, seen, users);

  mlir::Operation *init = nullptr;
  for (mlir::Operation *user : users) {
    if (!store(cell, user))
      continue;
    if (init)
      return user->emitOpError("async exception cell has multiple null "
                               "initialization stores");
    init = user;
  }

  if (!init)
    return owner->emitOpError(
        "async exception cell allocation lacks null initialization store");

  for (mlir::Operation *user : users) {
    if (user == init)
      continue;
    if (!dominance.dominates(init, user))
      return user->emitOpError("async exception cell is used before its null "
                               "initialization store");
  }
  return mlir::success();
}

} // namespace init

} // namespace exception_cell

namespace cancel_flag {

namespace init {

static bool store(mlir::Value flag, mlir::Operation *user) {
  auto store = mlir::dyn_cast<mlir::memref::StoreOp>(user);
  if (!store ||
      pointer::stripCasts(store.getMemref()) != pointer::stripCasts(flag))
    return false;
  if (store.getIndices().size() != 1)
    return false;
  auto index = constant::index(store.getIndices().front());
  return index && *index == 0 && constant::memrefInt(store.getValue(), 0);
}

static mlir::LogicalResult verify(mlir::Value flag, mlir::Operation *owner,
                                  mlir::DominanceInfo &dominance) {
  mlir::Operation *init = nullptr;
  for (mlir::Operation *user : flag.getUsers()) {
    if (!store(flag, user))
      continue;
    if (init)
      return user->emitOpError(
          "async cancel flag has multiple zero initialization stores");
    init = user;
  }

  if (!init)
    return owner->emitOpError(
        "async cancel flag allocation lacks zero initialization store");

  for (mlir::Operation *user : flag.getUsers()) {
    if (user == init)
      continue;
    if (!dominance.dominates(init, user))
      return user->emitOpError("async cancel flag is used before its zero "
                               "initialization store");
  }
  return mlir::success();
}

} // namespace init

} // namespace cancel_flag

mlir::LogicalResult
verifier::async_runtime::Cells::verify(mlir::Operation *funcLike) {
  mlir::DominanceInfo dominance(funcLike);
  bool failedAny = false;

  funcLike->walk([&](mlir::LLVM::CallOp call) {
    if (!call->hasAttr(AsyncSafetyAttrs::kExceptionCell))
      return;
    if (call->getNumResults() != 1)
      return;
    if (mlir::failed(exception_cell::init::verify(call.getResult(), dominance,
                                                  call.getOperation())))
      failedAny = true;
  });

  funcLike->walk([&](mlir::memref::AllocOp alloc) {
    if (!alloc->hasAttr(AsyncSafetyAttrs::kCancelFlag))
      return;
    if (mlir::failed(cancel_flag::init::verify(
            alloc.getResult(), alloc.getOperation(), dominance)))
      failedAny = true;
  });
  funcLike->walk([&](mlir::memref::AllocaOp alloca) {
    if (!alloca->hasAttr(AsyncSafetyAttrs::kCancelFlag))
      return;
    if (mlir::failed(cancel_flag::init::verify(
            alloca.getResult(), alloca.getOperation(), dominance)))
      failedAny = true;
  });

  return mlir::failure(failedAny);
}

} // namespace py::threadsafe
