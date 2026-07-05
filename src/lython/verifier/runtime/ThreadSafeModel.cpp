#include "runtime/ThreadSafeModel.h"

#include "Ownership.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringSwitch.h"

#include <optional>

namespace py::threadsafe {
namespace {

namespace own = py::ownership;

bool isOperationName(mlir::Operation *op, llvm::StringRef name) {
  return op && op->getName().getStringRef() == name;
}

std::optional<std::int64_t> constantIntValue(mlir::Value value) {
  auto constant = value ? value.getDefiningOp<mlir::arith::ConstantOp>()
                        : nullptr;
  if (!constant)
    return std::nullopt;
  auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
  if (!integer)
    return std::nullopt;
  return integer.getValue().getSExtValue();
}

bool isEntryBlockArgument(mlir::Value value) {
  auto argument = mlir::dyn_cast_if_present<mlir::BlockArgument>(value);
  if (!argument)
    return false;
  auto function =
      mlir::dyn_cast_or_null<mlir::func::FuncOp>(
          argument.getOwner()->getParentOp());
  return function && !function.empty() &&
         argument.getOwner() == &function.getBody().front();
}

bool isCapturedBlockArgument(mlir::Value value) {
  auto argument = mlir::dyn_cast_if_present<mlir::BlockArgument>(value);
  return argument && !isEntryBlockArgument(value);
}

bool isOwnedLocalObject(mlir::Value value) {
  mlir::Operation *def = value ? value.getDefiningOp() : nullptr;
  return def && def->hasAttr(own::kOwnedLocalObjectAttr);
}

bool isOwnedCallResult(mlir::Value value) {
  auto result = mlir::dyn_cast_if_present<mlir::OpResult>(value);
  if (!result)
    return false;
  auto call = mlir::dyn_cast_or_null<mlir::func::CallOp>(result.getOwner());
  if (!call)
    return false;
  mlir::ModuleOp module = call->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return false;
  mlir::func::FuncOp callee =
      module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
  return callee && own::functionOwnsResultAt(callee, result.getResultNumber());
}

} // namespace

std::optional<AtomicOrderingRank> parseAtomicOrdering(llvm::StringRef value) {
  return llvm::StringSwitch<std::optional<AtomicOrderingRank>>(value)
      .Case("not_atomic", AtomicOrderingRank::NotAtomic)
      .Case("unordered", AtomicOrderingRank::Unordered)
      .Case("monotonic", AtomicOrderingRank::Monotonic)
      .Case("acquire", AtomicOrderingRank::Acquire)
      .Case("release", AtomicOrderingRank::Release)
      .Case("acq_rel", AtomicOrderingRank::AcqRel)
      .Case("seq_cst", AtomicOrderingRank::SeqCst)
      .Default(std::nullopt);
}

llvm::StringRef atomicOrderingName(AtomicOrderingRank ordering) {
  switch (ordering) {
  case AtomicOrderingRank::NotAtomic:
    return "not_atomic";
  case AtomicOrderingRank::Unordered:
    return "unordered";
  case AtomicOrderingRank::Monotonic:
    return "monotonic";
  case AtomicOrderingRank::Acquire:
    return "acquire";
  case AtomicOrderingRank::Release:
    return "release";
  case AtomicOrderingRank::AcqRel:
    return "acq_rel";
  case AtomicOrderingRank::SeqCst:
    return "seq_cst";
  case AtomicOrderingRank::Invalid:
    return "<invalid>";
  }
  return "<invalid>";
}

bool orderingAtLeast(AtomicOrderingRank actual, AtomicOrderingRank required) {
  switch (required) {
  case AtomicOrderingRank::NotAtomic:
    return actual != AtomicOrderingRank::Invalid;
  case AtomicOrderingRank::Unordered:
    return actual == AtomicOrderingRank::Unordered ||
           actual == AtomicOrderingRank::Monotonic ||
           actual == AtomicOrderingRank::Acquire ||
           actual == AtomicOrderingRank::Release ||
           actual == AtomicOrderingRank::AcqRel ||
           actual == AtomicOrderingRank::SeqCst;
  case AtomicOrderingRank::Monotonic:
    return actual == AtomicOrderingRank::Monotonic ||
           actual == AtomicOrderingRank::Acquire ||
           actual == AtomicOrderingRank::Release ||
           actual == AtomicOrderingRank::AcqRel ||
           actual == AtomicOrderingRank::SeqCst;
  case AtomicOrderingRank::Acquire:
    return actual == AtomicOrderingRank::Acquire ||
           actual == AtomicOrderingRank::AcqRel ||
           actual == AtomicOrderingRank::SeqCst;
  case AtomicOrderingRank::Release:
    return actual == AtomicOrderingRank::Release ||
           actual == AtomicOrderingRank::AcqRel ||
           actual == AtomicOrderingRank::SeqCst;
  case AtomicOrderingRank::AcqRel:
    return actual == AtomicOrderingRank::AcqRel ||
           actual == AtomicOrderingRank::SeqCst;
  case AtomicOrderingRank::SeqCst:
    return actual == AtomicOrderingRank::SeqCst;
  case AtomicOrderingRank::Invalid:
    return false;
  }
  return false;
}

std::optional<RetainPremiseKind>
parseRetainPremise(llvm::StringRef value) {
  return llvm::StringSwitch<std::optional<RetainPremiseKind>>(value)
      .Case("", RetainPremiseKind::None)
      .Case("owned-token", RetainPremiseKind::OwnedToken)
      .Case("entry-borrowed", RetainPremiseKind::EntryBorrowed)
      .Case("captured-borrowed", RetainPremiseKind::CapturedBorrowed)
      .Case("aggregate-borrow", RetainPremiseKind::AggregateBorrow)
      .Case("locked-borrow", RetainPremiseKind::LockedBorrow)
      .Default(std::nullopt);
}

std::optional<RetainPremiseSourceKind>
parseRetainPremiseSource(llvm::StringRef value) {
  return llvm::StringSwitch<std::optional<RetainPremiseSourceKind>>(value)
      .Case("", RetainPremiseSourceKind::Unknown)
      .Case("unknown", RetainPremiseSourceKind::Unknown)
      .Case("entry-argument", RetainPremiseSourceKind::EntryArgument)
      .Case("captured-argument", RetainPremiseSourceKind::CapturedArgument)
      .Case("owned-local-object", RetainPremiseSourceKind::OwnedLocalObject)
      .Case("owned-call-result", RetainPremiseSourceKind::OwnedCallResult)
      .Case("aggregate-borrow", RetainPremiseSourceKind::AggregateBorrow)
      .Case("locked-borrow", RetainPremiseSourceKind::LockedBorrow)
      .Default(std::nullopt);
}

llvm::StringRef retainPremiseName(RetainPremiseKind premise) {
  switch (premise) {
  case RetainPremiseKind::None:
    return "";
  case RetainPremiseKind::OwnedToken:
    return "owned-token";
  case RetainPremiseKind::EntryBorrowed:
    return "entry-borrowed";
  case RetainPremiseKind::CapturedBorrowed:
    return "captured-borrowed";
  case RetainPremiseKind::AggregateBorrow:
    return "aggregate-borrow";
  case RetainPremiseKind::LockedBorrow:
    return "locked-borrow";
  }
  return "<invalid>";
}

llvm::StringRef retainPremiseSourceName(RetainPremiseSourceKind source) {
  switch (source) {
  case RetainPremiseSourceKind::Unknown:
    return "unknown";
  case RetainPremiseSourceKind::EntryArgument:
    return "entry-argument";
  case RetainPremiseSourceKind::CapturedArgument:
    return "captured-argument";
  case RetainPremiseSourceKind::OwnedLocalObject:
    return "owned-local-object";
  case RetainPremiseSourceKind::OwnedCallResult:
    return "owned-call-result";
  case RetainPremiseSourceKind::AggregateBorrow:
    return "aggregate-borrow";
  case RetainPremiseSourceKind::LockedBorrow:
    return "locked-borrow";
  }
  return "unknown";
}

AtomicOperationKind classifyAtomicOperation(mlir::Operation *op) {
  if (isOperationName(op, "memref.load"))
    return AtomicOperationKind::Load;
  if (isOperationName(op, "memref.store"))
    return AtomicOperationKind::Store;
  if (isOperationName(op, "memref.generic_atomic_rmw"))
    return AtomicOperationKind::RMW;
  return AtomicOperationKind::Unsupported;
}

AtomicRoleKind classifyAtomicRole(llvm::StringRef role,
                                  AtomicOperationKind operationKind) {
  if (role.contains("refcount.retain"))
    return AtomicRoleKind::RefcountRetain;
  if (role.contains("refcount.release"))
    return AtomicRoleKind::RefcountRelease;
  if (role.contains("refcount.load"))
    return AtomicRoleKind::RefcountLoad;
  if (role.contains(".load"))
    return AtomicRoleKind::SharedLoad;
  if (operationKind == AtomicOperationKind::Store)
    return AtomicRoleKind::SharedStore;
  if (operationKind == AtomicOperationKind::RMW)
    return AtomicRoleKind::SynchronizingRMW;
  return AtomicRoleKind::Unknown;
}

AtomicOrderingRank requiredOrdering(AtomicRoleKind roleKind,
                                    AtomicOperationKind operationKind) {
  switch (roleKind) {
  case AtomicRoleKind::RefcountRetain:
    return AtomicOrderingRank::Monotonic;
  case AtomicRoleKind::RefcountRelease:
    return AtomicOrderingRank::AcqRel;
  case AtomicRoleKind::RefcountLoad:
  case AtomicRoleKind::SharedLoad:
    return AtomicOrderingRank::Acquire;
  case AtomicRoleKind::SharedStore:
    return AtomicOrderingRank::Release;
  case AtomicRoleKind::SynchronizingRMW:
    return AtomicOrderingRank::AcqRel;
  case AtomicRoleKind::Unknown:
    break;
  }
  if (operationKind == AtomicOperationKind::RMW)
    return AtomicOrderingRank::AcqRel;
  return AtomicOrderingRank::Invalid;
}

llvm::SmallVector<HappensBeforeEdge, 4>
schedulerHappensBeforeEdges(const AtomicContract &contract) {
  llvm::SmallVector<HappensBeforeEdge, 4> edges;
  llvm::StringRef role(contract.role);
  if (role.contains("refcount"))
    return edges;

  auto publish = [&](llvm::StringRef resource) {
    edges.push_back({resource.str(), HappensBeforeEffect::Publish});
  };
  auto acquire = [&](llvm::StringRef resource) {
    edges.push_back({resource.str(), HappensBeforeEffect::Acquire});
  };

  if (contract.operationKind == AtomicOperationKind::Load) {
    if (role == "asyncio.ready.tail.load")
      acquire("asyncio.ready.tail");
    else if (role == "asyncio.loop.running.load")
      acquire("asyncio.loop.running");
    else if (role == "asyncio.future.state.load")
      acquire("asyncio.future.state");
    else if (role == "asyncio.task.coroutine.target.load")
      acquire("coroutine.target");
    else if (role == "asyncio.task.state.load")
      acquire("asyncio.task.state");
    else if (role == "asyncio.task.cancel.requests.load")
      acquire("asyncio.task.cancel.requests");
    return edges;
  }

  if (role == "coroutine.target.publish")
    publish("coroutine.target");
  else if (role.starts_with("coroutine.state."))
    publish("coroutine.state");
  else if (role == "asyncio.loop.running.publish")
    publish("asyncio.loop.running");
  else if (role == "asyncio.ready.tail.publish" ||
           role == "asyncio.ready.enqueue")
    publish("asyncio.ready.tail");
  else if (role == "asyncio.ready.pop")
    publish("asyncio.ready.head");
  else if (role == "asyncio.timer.record" ||
           role == "asyncio.timer.dispatch_due")
    publish("asyncio.timer.count");
  else if (role == "asyncio.future.state.publish" ||
           role == "asyncio.future.cancel" ||
           role == "asyncio.future.finish.reserve" ||
           role == "asyncio.future.finish")
    publish("asyncio.future.state");
  else if (role == "asyncio.future.cancel.requests")
    publish("asyncio.future.cancel.requests");
  else if (role == "asyncio.future.result.token" ||
           role == "asyncio.future.result.token.clear" ||
           role == "asyncio.future.exception.token" ||
           role == "asyncio.future.exception.token.clear")
    publish("asyncio.future.payload");
  else if (role == "asyncio.future.callback.record")
    publish("asyncio.future.callbacks");
  else if (role == "asyncio.task.state.publish" ||
           role == "asyncio.task.cancel" ||
           role == "asyncio.task.resume.begin" ||
           role == "asyncio.task.resume.complete" ||
           role == "asyncio.task.finish")
    publish("asyncio.task.state");
  else if (role == "asyncio.task.cancel.requests.publish" ||
           role == "asyncio.task.cancel.requests" ||
           role == "asyncio.task.uncancel")
    publish("asyncio.task.cancel.requests");
  else if (role == "asyncio.task.callback.record")
    publish("asyncio.task.callbacks");

  return edges;
}

std::optional<std::int64_t> expectedAtomicSlot(llvm::StringRef role) {
  if (role.contains("refcount"))
    return 0;
  return llvm::StringSwitch<std::optional<std::int64_t>>(role)
      .Case("coroutine.target.publish", 3)
      .Case("asyncio.task.coroutine.target.load", 3)
      .Case("coroutine.state.resume_begin", 2)
      .Case("coroutine.state.resume_complete", 2)
      .Case("coroutine.state.resume_suspend", 2)
      .Case("coroutine.state.close", 2)
      .Case("asyncio.loop.running.publish", 2)
      .Case("asyncio.loop.running.load", 2)
      .Case("asyncio.loop.stop", 3)
      .Case("asyncio.ready.tail.publish", 5)
      .Case("asyncio.ready.enqueue", 5)
      .Case("asyncio.ready.tail.load", 5)
      .Case("asyncio.ready.pop", 4)
      .Case("asyncio.timer.record", 6)
      .Case("asyncio.timer.dispatch_due", 6)
      .Case("asyncio.future.state.publish", 2)
      .Case("asyncio.future.state.load", 2)
      .Case("asyncio.future.cancel", 2)
      .Case("asyncio.future.cancel.requests", 3)
      .Case("asyncio.future.finish.reserve", 2)
      .Case("asyncio.future.finish", 2)
      .Case("asyncio.future.result.token", 7)
      .Case("asyncio.future.result.token.clear", 7)
      .Case("asyncio.future.exception.token", 8)
      .Case("asyncio.future.exception.token.clear", 8)
      .Case("asyncio.future.callback.record", 4)
      .Case("asyncio.task.state.publish", 2)
      .Case("asyncio.task.state.load", 2)
      .Case("asyncio.task.cancel", 2)
      .Case("asyncio.task.cancel.requests.publish", 3)
      .Case("asyncio.task.cancel.requests", 3)
      .Case("asyncio.task.cancel.requests.load", 3)
      .Case("asyncio.task.uncancel", 3)
      .Case("asyncio.task.resume.begin", 2)
      .Case("asyncio.task.resume.complete", 2)
      .Case("asyncio.task.finish", 2)
      .Case("asyncio.task.callback.record", 4)
      .Default(std::nullopt);
}

mlir::FailureOr<std::optional<AtomicContract>>
readAtomicContract(mlir::Operation *op) {
  auto role = op->getAttrOfType<mlir::StringAttr>(own::kAtomicRoleAttr);
  auto ordering =
      op->getAttrOfType<mlir::StringAttr>(own::kAtomicOrderingAttr);
  if (!role && !ordering)
    return std::optional<AtomicContract>();
  if (!role || !ordering)
    return op->emitError() << "atomic contracts require both "
                           << own::kAtomicRoleAttr << " and "
                           << own::kAtomicOrderingAttr;

  std::optional<AtomicOrderingRank> orderingRank =
      parseAtomicOrdering(ordering.getValue());
  if (!orderingRank)
    return op->emitError() << "unknown atomic ordering "
                           << ordering.getValue();

  AtomicContract contract;
  contract.role = role.getValue().str();
  contract.ordering = ordering.getValue().str();
  contract.orderingRank = *orderingRank;
  contract.operationKind = classifyAtomicOperation(op);
  contract.roleKind = classifyAtomicRole(role.getValue(), contract.operationKind);

  auto retainPremise =
      op->getAttrOfType<mlir::StringAttr>(own::kAtomicRetainPremiseAttr);
  contract.retainPremise =
      retainPremise ? retainPremise.getValue().str() : std::string();
  std::optional<RetainPremiseKind> premise =
      parseRetainPremise(contract.retainPremise);
  if (!premise)
    return op->emitError() << "unknown atomic retain premise "
                           << contract.retainPremise;
  contract.retainPremiseKind = *premise;
  return std::optional<AtomicContract>(std::move(contract));
}

mlir::LogicalResult verifyAtomicContractShape(mlir::Operation *op,
                                              const AtomicContract &contract) {
  if (contract.operationKind == AtomicOperationKind::Unsupported)
    return op->emitError()
           << "atomic role must be attached to memref load/store/RMW operation";

  if ((contract.roleKind == AtomicRoleKind::RefcountRetain ||
       contract.roleKind == AtomicRoleKind::RefcountRelease) &&
      contract.operationKind != AtomicOperationKind::RMW)
    return op->emitError()
           << "refcount atomic role must be attached to "
              "memref.generic_atomic_rmw";

  if ((contract.roleKind == AtomicRoleKind::RefcountLoad ||
       contract.roleKind == AtomicRoleKind::SharedLoad) &&
      contract.operationKind != AtomicOperationKind::Load)
    return op->emitError() << "load atomic role must be on memref.load";

  AtomicOrderingRank required =
      requiredOrdering(contract.roleKind, contract.operationKind);
  if (required == AtomicOrderingRank::Invalid)
    return op->emitError() << "atomic role " << contract.role
                           << " has no no-GIL ordering rule";
  if (!orderingAtLeast(contract.orderingRank, required))
    return op->emitError()
           << "atomic role " << contract.role << " requires at least "
           << atomicOrderingName(required) << " ordering";

  if (contract.roleKind == AtomicRoleKind::RefcountRetain &&
      contract.retainPremiseKind == RetainPremiseKind::None)
    return op->emitError() << "refcount retain requires a valid "
                           << own::kAtomicRetainPremiseAttr;
  if (contract.roleKind != AtomicRoleKind::RefcountRetain &&
      contract.retainPremiseKind != RetainPremiseKind::None)
    return op->emitError() << own::kAtomicRetainPremiseAttr
                           << " is only valid on refcount retain RMW";

  return mlir::success();
}

mlir::Value atomicMemoryBase(mlir::Operation *op) {
  if (!op || op->getNumOperands() == 0)
    return {};
  if (auto rmw = mlir::dyn_cast<mlir::memref::GenericAtomicRMWOp>(op))
    return rmw.getMemref();
  if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op))
    return load.getMemRef();
  if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op))
    return store.getMemRef();
  return op->getOperand(0);
}

std::optional<std::int64_t> atomicSlotIndex(mlir::Operation *op) {
  mlir::ValueRange indices;
  if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
    indices = load.getIndices();
  } else if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
    indices = store.getIndices();
  } else if (auto rmw =
                 mlir::dyn_cast<mlir::memref::GenericAtomicRMWOp>(op)) {
    indices = rmw.getIndices();
  } else {
    return std::nullopt;
  }
  if (indices.size() != 1)
    return std::nullopt;
  return constantIntValue(indices.front());
}

mlir::Value stripThreadSafeView(mlir::Value value) {
  while (value) {
    mlir::Operation *def = value.getDefiningOp();
    if (!def || def->getNumOperands() == 0 || def->getNumResults() != 1)
      return value;
    llvm::StringRef name = def->getName().getStringRef();
    if (name != "memref.cast" && name != "memref.subview" &&
        name != "memref.reinterpret_cast")
      return value;
    value = def->getOperand(0);
  }
  return value;
}

mlir::FailureOr<RetainPremiseInference>
inferRetainPremiseSource(mlir::Operation *op,
                         const AtomicContract &contract) {
  if (contract.roleKind != AtomicRoleKind::RefcountRetain)
    return op->emitError()
           << "retain premise inference requires a refcount retain role";

  RetainPremiseInference inference;
  inference.root = stripThreadSafeView(atomicMemoryBase(op));
  if (!inference.root)
    return op->emitError() << "refcount retain has no memory root";

  if (isEntryBlockArgument(inference.root)) {
    inference.source = RetainPremiseSourceKind::EntryArgument;
    return inference;
  }
  if (isCapturedBlockArgument(inference.root)) {
    inference.source = RetainPremiseSourceKind::CapturedArgument;
    return inference;
  }
  if (isOwnedLocalObject(inference.root)) {
    inference.source = RetainPremiseSourceKind::OwnedLocalObject;
    return inference;
  }
  if (isOwnedCallResult(inference.root)) {
    inference.source = RetainPremiseSourceKind::OwnedCallResult;
    return inference;
  }

  return inference;
}

bool retainPremiseAllowsSource(RetainPremiseKind premise,
                               RetainPremiseSourceKind source) {
  switch (premise) {
  case RetainPremiseKind::OwnedToken:
    return source == RetainPremiseSourceKind::OwnedLocalObject ||
           source == RetainPremiseSourceKind::OwnedCallResult;
  case RetainPremiseKind::EntryBorrowed:
    return source == RetainPremiseSourceKind::EntryArgument;
  case RetainPremiseKind::CapturedBorrowed:
    return source == RetainPremiseSourceKind::CapturedArgument;
  case RetainPremiseKind::AggregateBorrow:
    return source == RetainPremiseSourceKind::AggregateBorrow;
  case RetainPremiseKind::LockedBorrow:
    return source == RetainPremiseSourceKind::LockedBorrow;
  case RetainPremiseKind::None:
    return false;
  }
  return false;
}

mlir::LogicalResult verifyRetainPremise(mlir::Operation *op,
                                        const AtomicContract &contract) {
  if (contract.roleKind != AtomicRoleKind::RefcountRetain)
    return mlir::success();
  mlir::FailureOr<RetainPremiseInference> inferred =
      inferRetainPremiseSource(op, contract);
  if (mlir::failed(inferred))
    return mlir::failure();
  if (!retainPremiseAllowsSource(contract.retainPremiseKind,
                                 inferred->source))
    return op->emitError()
           << "refcount retain premise " << contract.retainPremise
           << " does not match inferred live token source "
           << retainPremiseSourceName(inferred->source);
  return mlir::success();
}

} // namespace py::threadsafe
