#include "runtime/Detail.h"
#include "runtime/Verification.h"

#include "Ownership.h"
#include "runtime/ThreadSafeModel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace py::lowering {
namespace {

namespace own = py::ownership;
namespace ts = py::threadsafe;

inline constexpr llvm::StringLiteral kLoweredSafetyContractsAttr{
    "ly.lowered_safety_contracts"};

std::optional<int64_t> constantIntValue(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
  if (!integer)
    return std::nullopt;
  return integer.getValue().getSExtValue();
}

bool isConstant(mlir::Value value, int64_t expected) {
  std::optional<int64_t> actual = constantIntValue(value);
  return actual && *actual == expected;
}

bool opUsesCurrentAndConstant(mlir::Operation *op, llvm::StringRef opName,
                              mlir::Value current, int64_t constant) {
  if (op->getName().getStringRef() != opName || op->getNumOperands() != 2)
    return false;
  return (op->getOperand(0) == current &&
          isConstant(op->getOperand(1), constant)) ||
         (op->getOperand(1) == current &&
          isConstant(op->getOperand(0), constant));
}

bool hasPositiveCurrentCheck(mlir::Region &region, mlir::Value current) {
  bool found = false;
  region.walk([&](mlir::arith::CmpIOp cmp) {
    if (cmp.getPredicate() != mlir::arith::CmpIPredicate::sgt)
      return;
    if (cmp.getLhs() == current && isConstant(cmp.getRhs(), 0))
      found = true;
  });
  return found;
}

bool hasImmortalCheck(mlir::Region &region, mlir::Value current) {
  bool found = false;
  region.walk([&](mlir::arith::CmpIOp cmp) {
    if (cmp.getPredicate() != mlir::arith::CmpIPredicate::eq)
      return;
    constexpr int64_t kImmortal = 9223372036854775807LL;
    if ((cmp.getLhs() == current && isConstant(cmp.getRhs(), kImmortal)) ||
        (cmp.getRhs() == current && isConstant(cmp.getLhs(), kImmortal)))
      found = true;
  });
  return found;
}

mlir::Operation *findRefcountDelta(mlir::Region &region, mlir::Value current,
                                   bool retain) {
  mlir::Operation *found = nullptr;
  llvm::StringRef opName = retain ? "arith.addi" : "arith.subi";
  region.walk([&](mlir::Operation *op) {
    if (found)
      return;
    if (retain) {
      if (opUsesCurrentAndConstant(op, opName, current, 1))
        found = op;
      return;
    }
    if (op->getName().getStringRef() != opName || op->getNumOperands() != 2)
      return;
    if (op->getOperand(0) == current && isConstant(op->getOperand(1), 1))
      found = op;
  });
  return found;
}

bool hasRefcountDelta(mlir::Region &region, mlir::Value current, bool retain) {
  return findRefcountDelta(region, current, retain) != nullptr;
}

bool hasOppositeRefcountDelta(mlir::Region &region, mlir::Value current,
                              bool retain) {
  bool found = false;
  llvm::StringRef opposite = retain ? "arith.subi" : "arith.addi";
  region.walk([&](mlir::Operation *op) {
    if (found)
      return;
    if (opUsesCurrentAndConstant(op, opposite, current, 1))
      found = true;
  });
  return found;
}

bool valueDependsOnOperation(mlir::Value value, mlir::Operation *dependency,
                             unsigned depth = 0) {
  if (!value || !dependency || depth > 32)
    return false;
  if (value.getDefiningOp() == dependency)
    return true;
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  return llvm::any_of(def->getOperands(), [&](mlir::Value operand) {
    return valueDependsOnOperation(operand, dependency, depth + 1);
  });
}

bool atomicYieldDependsOnDelta(mlir::Region &region,
                               mlir::Operation *delta) {
  bool found = false;
  region.walk([&](mlir::Operation *op) {
    if (found || op->getName().getStringRef() != "memref.atomic_yield" ||
        op->getNumOperands() != 1)
      return;
    if (valueDependsOnOperation(op->getOperand(0), delta))
      found = true;
  });
  return found;
}

mlir::LogicalResult verifyRefcountAtomicBody(mlir::Operation *op, bool retain) {
  if (op->getName().getStringRef() != "memref.generic_atomic_rmw")
    return op->emitError() << "refcount atomic role must be attached to "
                           << "memref.generic_atomic_rmw";
  if (op->getNumRegions() != 1 || op->getRegion(0).empty() ||
      op->getRegion(0).front().getNumArguments() != 1)
    return op->emitError()
           << "refcount atomic region must expose current value";

  mlir::Region &region = op->getRegion(0);
  mlir::Value current = region.front().getArgument(0);
  if (!hasPositiveCurrentCheck(region, current))
    return op->emitError() << "refcount atomic body must prove current > 0";
  if (!hasImmortalCheck(region, current))
    return op->emitError()
           << "refcount atomic body must preserve immortal refcount";
  if (!hasRefcountDelta(region, current, retain))
    return op->emitError() << "refcount atomic body must apply "
                           << (retain ? "current + 1" : "current - 1");
  mlir::Operation *delta = findRefcountDelta(region, current, retain);
  if (!atomicYieldDependsOnDelta(region, delta))
    return op->emitError()
           << "refcount atomic body must yield a value derived from "
           << (retain ? "current + 1" : "current - 1");
  if (hasOppositeRefcountDelta(region, current, retain))
    return op->emitError()
           << "refcount atomic body mixes retain and release transitions";
  return mlir::success();
}

struct HappensBeforeResourceUse {
  mlir::Operation *op = nullptr;
};

ts::AtomicOperationKind operationKindFromName(llvm::StringRef name) {
  return llvm::StringSwitch<ts::AtomicOperationKind>(name)
      .Case("memref.load", ts::AtomicOperationKind::Load)
      .Case("memref.store", ts::AtomicOperationKind::Store)
      .Case("memref.generic_atomic_rmw", ts::AtomicOperationKind::RMW)
      .Default(ts::AtomicOperationKind::Unsupported);
}

bool isFuturePayloadRole(llvm::StringRef role) {
  return role == "asyncio.future.result.token" ||
         role == "asyncio.future.result.token.clear" ||
         role == "asyncio.future.exception.token" ||
         role == "asyncio.future.exception.token.clear";
}

void appendFunctionRoleTimeline(mlir::ModuleOp module, mlir::func::FuncOp func,
                                llvm::SmallVectorImpl<std::string> &timeline,
                                unsigned depth,
                                llvm::SmallPtrSetImpl<mlir::Operation *>
                                    &activeFunctions);

void appendOperationRoleTimeline(mlir::ModuleOp module, mlir::Operation *op,
                                 llvm::SmallVectorImpl<std::string> &timeline,
                                 unsigned depth,
                                 llvm::SmallPtrSetImpl<mlir::Operation *>
                                     &activeFunctions) {
  if (auto role =
          op->getAttrOfType<mlir::StringAttr>(own::kAtomicRoleAttr)) {
    timeline.push_back(role.getValue().str());
    return;
  }

  if (depth == 0)
    return;
  auto call = mlir::dyn_cast<mlir::func::CallOp>(op);
  if (!call)
    return;
  mlir::func::FuncOp callee =
      module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
  if (!callee)
    return;
  appendFunctionRoleTimeline(module, callee, timeline, depth - 1,
                             activeFunctions);
}

void appendFunctionRoleTimeline(mlir::ModuleOp module, mlir::func::FuncOp func,
                                llvm::SmallVectorImpl<std::string> &timeline,
                                unsigned depth,
                                llvm::SmallPtrSetImpl<mlir::Operation *>
                                    &activeFunctions) {
  if (!func || !activeFunctions.insert(func.getOperation()).second)
    return;
  func.walk([&](mlir::Operation *op) {
    if (op == func.getOperation())
      return;
    appendOperationRoleTimeline(module, op, timeline, depth, activeFunctions);
  });
  activeFunctions.erase(func.getOperation());
}

mlir::LogicalResult verifyAtomicContract(mlir::Operation *op) {
  mlir::FailureOr<std::optional<ts::AtomicContract>> contract =
      ts::readAtomicContract(op);
  if (mlir::failed(contract))
    return mlir::failure();
  if (!*contract)
    return mlir::success();
  if (mlir::failed(ts::verifyAtomicContractShape(op, **contract)))
    return mlir::failure();
  if (std::optional<std::int64_t> expected =
          ts::expectedAtomicSlot((*contract)->role)) {
    std::optional<std::int64_t> actual = ts::atomicSlotIndex(op);
    if (!actual)
      return op->emitError()
             << "atomic role " << (*contract)->role
             << " must use a single constant slot " << *expected;
    if (*actual != *expected)
      return op->emitError()
             << "atomic role " << (*contract)->role
             << " is attached to slot " << *actual << " but requires slot "
             << *expected;
  }
  if (mlir::failed(ts::verifyRetainPremise(op, **contract)))
    return mlir::failure();

  if ((*contract)->roleKind == ts::AtomicRoleKind::RefcountRetain)
    return verifyRefcountAtomicBody(op, /*retain=*/true);

  if ((*contract)->roleKind == ts::AtomicRoleKind::RefcountRelease)
    return verifyRefcountAtomicBody(op, /*retain=*/false);
  return mlir::success();
}

mlir::LogicalResult verifySchedulerHappensBeforeContracts(
    mlir::ModuleOp module) {
  llvm::StringSet<> published;
  llvm::StringMap<HappensBeforeResourceUse> firstAcquire;
  VerificationResult verified;

  module.walk([&](mlir::Operation *op) {
    if (verified.failed())
      return;
    mlir::FailureOr<std::optional<ts::AtomicContract>> contract =
        ts::readAtomicContract(op);
    if (mlir::failed(contract)) {
      verified.fail();
      return;
    }
    if (!*contract)
      return;
    for (const ts::HappensBeforeEdge &edge :
         ts::schedulerHappensBeforeEdges(**contract)) {
      if (edge.effect == ts::HappensBeforeEffect::Publish) {
        published.insert(edge.resource);
        continue;
      }
      if (!firstAcquire.contains(edge.resource))
        firstAcquire[edge.resource] = {op};
    }
  });
  if (verified.failed())
    return mlir::failure();

  for (const auto &entry : firstAcquire) {
    if (published.contains(entry.getKey()))
      continue;
    mlir::Operation *op = entry.getValue().op;
    return op->emitError()
           << "scheduler happens-before acquire for resource '"
           << entry.getKey() << "' has no matching runtime publisher";
  }

  return mlir::success();
}

mlir::LogicalResult verifyFuturePayloadStateOrdering(mlir::ModuleOp module) {
  VerificationResult verified;
  module.walk([&](mlir::func::FuncOp function) {
    if (verified.failed())
      return;
    llvm::SmallVector<std::string, 32> timeline;
    llvm::SmallPtrSet<mlir::Operation *, 8> activeFunctions;
    appendFunctionRoleTimeline(module, function, timeline, /*depth=*/4,
                               activeFunctions);

    std::optional<unsigned> firstPayload;
    std::optional<unsigned> lastPayload;
    std::optional<unsigned> firstReserve;
    for (auto [index, roleStorage] : llvm::enumerate(timeline)) {
      llvm::StringRef role(roleStorage);
      unsigned ordinal = static_cast<unsigned>(index);
      if (role == "asyncio.future.finish.reserve" && !firstReserve)
        firstReserve = ordinal;
      if (isFuturePayloadRole(role)) {
        if (!firstPayload)
          firstPayload = ordinal;
        lastPayload = ordinal;
      }
    }
    if (!firstPayload)
      return;

    if (!firstReserve || *firstReserve > *firstPayload) {
      verified.check(
          function.emitError()
          << "future payload publication must be guarded by an earlier "
             "asyncio.future.finish.reserve transition");
      return;
    }

    bool hasFinalPublishAfterPayload = false;
    for (auto [index, roleStorage] : llvm::enumerate(timeline)) {
      if (index <= *lastPayload)
        continue;
      if (roleStorage == "asyncio.future.finish") {
        hasFinalPublishAfterPayload = true;
        break;
      }
    }
    if (!hasFinalPublishAfterPayload)
      verified.check(
          function.emitError()
          << "future payload release stores must happen before the final "
             "asyncio.future.finish publication");
  });
  return verified.get();
}

struct PreservedSafetyContract {
  std::int64_t ordinal = -1;
  llvm::StringRef functionName;
  llvm::StringRef operationName;
  llvm::StringRef role;
  llvm::StringRef ordering;
  llvm::StringRef retainPremise;
  llvm::StringRef retainPremiseSource;
  std::int64_t slot = -1;
};

mlir::FailureOr<llvm::StringRef>
readStringField(mlir::ModuleOp module, mlir::DictionaryAttr entry,
                llvm::StringRef fieldName, unsigned index,
                bool required = true) {
  auto attr = mlir::dyn_cast_if_present<mlir::StringAttr>(entry.get(fieldName));
  if (attr)
    return attr.getValue();
  if (!required)
    return llvm::StringRef();
  return module.emitError()
         << kLoweredSafetyContractsAttr << " entry " << index
         << " must contain string field '" << fieldName << "'";
}

mlir::FailureOr<PreservedSafetyContract>
readPreservedSafetyContract(mlir::ModuleOp module, mlir::Attribute raw,
                            unsigned index) {
  auto entry = mlir::dyn_cast<mlir::DictionaryAttr>(raw);
  if (!entry)
    return module.emitError()
           << kLoweredSafetyContractsAttr << " entry " << index
           << " must be a dictionary";

  auto ordinalAttr =
      mlir::dyn_cast_if_present<mlir::IntegerAttr>(entry.get("ordinal"));
  if (!ordinalAttr)
    return module.emitError()
           << kLoweredSafetyContractsAttr << " entry " << index
           << " must contain integer field 'ordinal'";

  PreservedSafetyContract contract;
  contract.ordinal = ordinalAttr.getInt();
  if (contract.ordinal < 0)
    return module.emitError()
           << kLoweredSafetyContractsAttr << " entry " << index
           << " has negative ordinal " << contract.ordinal;

  auto functionName = readStringField(module, entry, "function", index);
  if (mlir::failed(functionName))
    return mlir::failure();
  contract.functionName = *functionName;

  auto operationName = readStringField(module, entry, "op", index);
  if (mlir::failed(operationName))
    return mlir::failure();
  contract.operationName = *operationName;

  auto role = readStringField(module, entry, "role", index);
  if (mlir::failed(role))
    return mlir::failure();
  contract.role = *role;
  if (contract.role.empty())
    return module.emitError()
           << kLoweredSafetyContractsAttr << " entry " << index
           << " has an empty atomic role";

  auto ordering = readStringField(module, entry, "ordering", index);
  if (mlir::failed(ordering))
    return mlir::failure();
  contract.ordering = *ordering;
  if (contract.ordering.empty())
    return module.emitError()
           << kLoweredSafetyContractsAttr << " entry " << index
           << " has an empty atomic ordering";

  auto retainPremise =
      readStringField(module, entry, "retain_premise", index,
                      /*required=*/false);
  if (mlir::failed(retainPremise))
    return mlir::failure();
  contract.retainPremise = *retainPremise;
  auto retainPremiseSource =
      readStringField(module, entry, "retain_premise_source", index,
                      /*required=*/false);
  if (mlir::failed(retainPremiseSource))
    return mlir::failure();
  contract.retainPremiseSource = *retainPremiseSource;

  if (auto slotAttr =
          mlir::dyn_cast_if_present<mlir::IntegerAttr>(entry.get("slot")))
    contract.slot = slotAttr.getInt();
  return contract;
}

mlir::LogicalResult
verifyPreservedSafetyContract(mlir::ModuleOp module,
                              const PreservedSafetyContract &contract,
                              unsigned index) {
  std::optional<ts::AtomicOrderingRank> ordering =
      ts::parseAtomicOrdering(contract.ordering);
  if (!ordering)
    return module.emitError()
           << kLoweredSafetyContractsAttr << " entry " << index
           << " has unknown atomic ordering " << contract.ordering;
  std::optional<ts::RetainPremiseKind> retainPremise =
      ts::parseRetainPremise(contract.retainPremise);
  if (!retainPremise)
    return module.emitError()
           << kLoweredSafetyContractsAttr << " entry " << index
           << " has unknown retain premise " << contract.retainPremise;
  std::optional<ts::RetainPremiseSourceKind> retainPremiseSource =
      ts::parseRetainPremiseSource(contract.retainPremiseSource);
  if (!retainPremiseSource)
    return module.emitError()
           << kLoweredSafetyContractsAttr << " entry " << index
           << " has unknown retain premise source "
           << contract.retainPremiseSource;

  bool isLoad = contract.operationName == "memref.load";
  bool isStore = contract.operationName == "memref.store";
  bool isRmw = contract.operationName == "memref.generic_atomic_rmw";

  if (std::optional<std::int64_t> expected =
          ts::expectedAtomicSlot(contract.role)) {
    if (contract.slot < 0)
      return module.emitError()
             << kLoweredSafetyContractsAttr << " entry " << index
             << " role " << contract.role
             << " requires preserved atomic slot " << *expected;
    if (contract.slot != *expected)
      return module.emitError()
             << kLoweredSafetyContractsAttr << " entry " << index
             << " role " << contract.role << " preserves slot "
             << contract.slot << " but requires slot " << *expected;
  }

  if (contract.role.contains("refcount.retain")) {
    if (!isRmw)
      return module.emitError()
             << kLoweredSafetyContractsAttr << " entry " << index
             << " refcount retain contract must originate from "
                "memref.generic_atomic_rmw";
    if (!ts::orderingAtLeast(*ordering, ts::AtomicOrderingRank::Monotonic))
      return module.emitError()
             << kLoweredSafetyContractsAttr << " entry " << index
             << " refcount retain requires monotonic ordering";
    if (*retainPremise == ts::RetainPremiseKind::None)
      return module.emitError()
             << kLoweredSafetyContractsAttr << " entry " << index
             << " refcount retain requires a preserved valid "
             << own::kAtomicRetainPremiseAttr;
    if (contract.retainPremiseSource.empty() ||
        contract.retainPremiseSource == "unknown")
      return module.emitError()
             << kLoweredSafetyContractsAttr << " entry " << index
             << " refcount retain requires a preserved inferred live-token "
                "source";
    if (!ts::retainPremiseAllowsSource(*retainPremise,
                                       *retainPremiseSource))
      return module.emitError()
             << kLoweredSafetyContractsAttr << " entry " << index
             << " refcount retain premise " << contract.retainPremise
             << " does not match preserved live-token source "
             << contract.retainPremiseSource;
    return mlir::success();
  }

  if (contract.role.contains("refcount.release")) {
    if (!isRmw)
      return module.emitError()
             << kLoweredSafetyContractsAttr << " entry " << index
             << " refcount release contract must originate from "
                "memref.generic_atomic_rmw";
    if (!ts::orderingAtLeast(*ordering, ts::AtomicOrderingRank::AcqRel))
      return module.emitError()
             << kLoweredSafetyContractsAttr << " entry " << index
             << " refcount release requires acq_rel ordering";
    return mlir::success();
  }

  if (contract.role.contains(".load")) {
    if (!isLoad)
      return module.emitError()
             << kLoweredSafetyContractsAttr << " entry " << index
             << " load atomic role must originate from memref.load";
    if (!ts::orderingAtLeast(*ordering, ts::AtomicOrderingRank::Acquire))
      return module.emitError()
             << kLoweredSafetyContractsAttr << " entry " << index
             << " shared load requires acquire ordering";
    return mlir::success();
  }

  if (isStore) {
    if (!ts::orderingAtLeast(*ordering, ts::AtomicOrderingRank::Release))
      return module.emitError()
             << kLoweredSafetyContractsAttr << " entry " << index
             << " shared store requires release ordering";
    return mlir::success();
  }

  if (isRmw) {
    if (!ts::orderingAtLeast(*ordering, ts::AtomicOrderingRank::AcqRel))
      return module.emitError()
             << kLoweredSafetyContractsAttr << " entry " << index
             << " synchronizing RMW requires acq_rel ordering";
    return mlir::success();
  }

  return module.emitError()
         << kLoweredSafetyContractsAttr << " entry " << index
         << " must originate from memref load/store/RMW operation";
}

mlir::LogicalResult
verifyPreservedLoweredSafetyContracts(mlir::ModuleOp module) {
  auto contractsAttr =
      module->getAttrOfType<mlir::ArrayAttr>(kLoweredSafetyContractsAttr);
  if (!contractsAttr)
    return mlir::success();

  std::optional<std::int64_t> previousOrdinal;
  llvm::StringSet<> published;
  llvm::StringMap<unsigned> firstAcquire;
  for (auto [index, raw] : llvm::enumerate(contractsAttr)) {
    auto contract =
        readPreservedSafetyContract(module, raw, static_cast<unsigned>(index));
    if (mlir::failed(contract))
      return mlir::failure();
    if (previousOrdinal && contract->ordinal <= *previousOrdinal)
      return module.emitError()
             << kLoweredSafetyContractsAttr << " entry " << index
             << " ordinal " << contract->ordinal
             << " is not strictly greater than previous ordinal "
             << *previousOrdinal;
    previousOrdinal = contract->ordinal;
    if (mlir::failed(verifyPreservedSafetyContract(
            module, *contract, static_cast<unsigned>(index))))
      return mlir::failure();

    ts::AtomicContract atomic;
    atomic.role = contract->role.str();
    atomic.operationKind = operationKindFromName(contract->operationName);
    for (const ts::HappensBeforeEdge &edge :
         ts::schedulerHappensBeforeEdges(atomic)) {
      if (edge.effect == ts::HappensBeforeEffect::Publish) {
        published.insert(edge.resource);
        continue;
      }
      if (!firstAcquire.contains(edge.resource))
        firstAcquire[edge.resource] = static_cast<unsigned>(index);
    }
  }

  for (const auto &entry : firstAcquire) {
    if (published.contains(entry.getKey()))
      continue;
    return module.emitError()
           << kLoweredSafetyContractsAttr << " entry "
           << entry.getValue()
           << " scheduler happens-before acquire for resource '"
           << entry.getKey()
           << "' has no preserved matching runtime publisher";
  }

  return mlir::success();
}

mlir::LogicalResult verifyThreadSafeContracts(mlir::ModuleOp module) {
  if (mlir::failed(walkVerifyOperations(module, verifyAtomicContract)))
    return mlir::failure();
  if (mlir::failed(verifySchedulerHappensBeforeContracts(module)))
    return mlir::failure();
  if (mlir::failed(verifyFuturePayloadStateOrdering(module)))
    return mlir::failure();
  return verifyPreservedLoweredSafetyContracts(module);
}

class LLVMThreadSafeVerifierPass
    : public mlir::PassWrapper<LLVMThreadSafeVerifierPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLVMThreadSafeVerifierPass)

  llvm::StringRef getArgument() const final {
    return "lython-llvm-thread-safety-verifier";
  }
  llvm::StringRef getDescription() const final {
    return "verify lowered atomic thread-safety contracts";
  }

  void runOnOperation() final {
    if (mlir::failed(verifyThreadSafeContracts(getOperation())))
      signalPassFailure();
  }
};

} // namespace
} // namespace py::lowering

namespace py {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLLVMThreadSafeVerifierPass() {
  return std::make_unique<lowering::LLVMThreadSafeVerifierPass>();
}

void collectLoweredSafetyContracts(mlir::ModuleOp module,
                                   LoweredSafetyContracts &contracts) {
  mlir::Builder builder(module.getContext());
  std::int64_t ordinal = static_cast<std::int64_t>(contracts.contracts.size());
  module.walk([&](mlir::Operation *op) {
    auto role =
        op->getAttrOfType<mlir::StringAttr>(ownership::kAtomicRoleAttr);
    auto ordering = op->getAttrOfType<mlir::StringAttr>(
        ownership::kAtomicOrderingAttr);
    if (!role && !ordering)
      return;
    auto retainPremise = op->getAttrOfType<mlir::StringAttr>(
        ownership::kAtomicRetainPremiseAttr);
    LoweredSafetyContract contract;
    if (auto function = op->getParentOfType<mlir::func::FuncOp>())
      contract.functionName = function.getSymName().str();
    contract.operationName = op->getName().getStringRef().str();
    contract.role = role ? role.getValue().str() : std::string();
    contract.ordering = ordering ? ordering.getValue().str() : std::string();
    contract.retainPremise =
        retainPremise ? retainPremise.getValue().str() : std::string();
    if (std::optional<std::int64_t> slot =
            threadsafe::atomicSlotIndex(op))
      contract.slot = *slot;
    mlir::FailureOr<std::optional<threadsafe::AtomicContract>>
        atomic = threadsafe::readAtomicContract(op);
    if (mlir::succeeded(atomic) && *atomic &&
        (*atomic)->roleKind ==
            threadsafe::AtomicRoleKind::RefcountRetain) {
      mlir::FailureOr<threadsafe::RetainPremiseInference>
          inferred =
              threadsafe::inferRetainPremiseSource(op, **atomic);
      if (mlir::succeeded(inferred)) {
        contract.retainPremiseSource =
            threadsafe::retainPremiseSourceName(inferred->source)
                .str();
      }
    }
    contract.ordinal = ordinal++;
    op->setAttr("ly.lowered_safety_contract_id",
                builder.getI64IntegerAttr(contract.ordinal));
    contracts.contracts.push_back(std::move(contract));
  });
}

void collectLoweredSafetyContracts(mlir::ModuleOp module,
                                   const PyLLVMTypeConverter &,
                                   LoweredSafetyContracts &contracts) {
  collectLoweredSafetyContracts(module, contracts);
}

mlir::LogicalResult
preserveLoweredSafetyContracts(mlir::ModuleOp module,
                               const LoweredSafetyContracts &contracts) {
  mlir::Builder builder(module.getContext());
  llvm::SmallVector<mlir::Attribute, 32> entries;
  entries.reserve(contracts.contracts.size());
  for (const LoweredSafetyContract &contract : contracts.contracts) {
    llvm::SmallVector<mlir::NamedAttribute, 8> attrs;
    attrs.push_back(builder.getNamedAttr(
        "ordinal", builder.getI64IntegerAttr(contract.ordinal)));
    attrs.push_back(
        builder.getNamedAttr("slot", builder.getI64IntegerAttr(contract.slot)));
    attrs.push_back(builder.getNamedAttr(
        "function", builder.getStringAttr(contract.functionName)));
    attrs.push_back(builder.getNamedAttr(
        "op", builder.getStringAttr(contract.operationName)));
    attrs.push_back(
        builder.getNamedAttr("role", builder.getStringAttr(contract.role)));
    attrs.push_back(builder.getNamedAttr(
        "ordering", builder.getStringAttr(contract.ordering)));
    attrs.push_back(builder.getNamedAttr(
        "retain_premise", builder.getStringAttr(contract.retainPremise)));
    attrs.push_back(builder.getNamedAttr(
        "retain_premise_source",
        builder.getStringAttr(contract.retainPremiseSource)));
    entries.push_back(builder.getDictionaryAttr(attrs));
  }
  module->setAttr(lowering::kLoweredSafetyContractsAttr,
                  builder.getArrayAttr(entries));
  return mlir::success();
}

} // namespace py
