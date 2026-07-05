#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace py::threadsafe {

enum class AtomicOrderingRank {
  Invalid = -1,
  NotAtomic = 0,
  Unordered = 1,
  Monotonic = 2,
  Acquire = 3,
  Release = 4,
  AcqRel = 5,
  SeqCst = 6,
};

enum class AtomicOperationKind { Unsupported, Load, Store, RMW };

enum class AtomicRoleKind {
  Unknown,
  RefcountLoad,
  RefcountRetain,
  RefcountRelease,
  SharedLoad,
  SharedStore,
  SynchronizingRMW,
};

enum class RetainPremiseKind {
  None,
  OwnedToken,
  EntryBorrowed,
  CapturedBorrowed,
  AggregateBorrow,
  LockedBorrow,
};

enum class RetainPremiseSourceKind {
  Unknown,
  EntryArgument,
  CapturedArgument,
  OwnedLocalObject,
  OwnedCallResult,
  AggregateBorrow,
  LockedBorrow,
};

enum class HappensBeforeEffect {
  Publish,
  Acquire,
};

struct AtomicContract {
  std::string role;
  std::string ordering;
  std::string retainPremise;
  AtomicOrderingRank orderingRank = AtomicOrderingRank::Invalid;
  AtomicOperationKind operationKind = AtomicOperationKind::Unsupported;
  AtomicRoleKind roleKind = AtomicRoleKind::Unknown;
  RetainPremiseKind retainPremiseKind = RetainPremiseKind::None;
};

struct HappensBeforeEdge {
  std::string resource;
  HappensBeforeEffect effect = HappensBeforeEffect::Publish;
};

struct RetainPremiseInference {
  RetainPremiseSourceKind source = RetainPremiseSourceKind::Unknown;
  mlir::Value root;
};

std::optional<AtomicOrderingRank> parseAtomicOrdering(llvm::StringRef value);
llvm::StringRef atomicOrderingName(AtomicOrderingRank ordering);
bool orderingAtLeast(AtomicOrderingRank actual, AtomicOrderingRank required);

std::optional<RetainPremiseKind>
parseRetainPremise(llvm::StringRef value);
std::optional<RetainPremiseSourceKind>
parseRetainPremiseSource(llvm::StringRef value);
llvm::StringRef retainPremiseName(RetainPremiseKind premise);
llvm::StringRef retainPremiseSourceName(RetainPremiseSourceKind source);

AtomicOperationKind classifyAtomicOperation(mlir::Operation *op);
AtomicRoleKind classifyAtomicRole(llvm::StringRef role,
                                  AtomicOperationKind operationKind);
AtomicOrderingRank requiredOrdering(AtomicRoleKind roleKind,
                                    AtomicOperationKind operationKind);
llvm::SmallVector<HappensBeforeEdge, 4>
schedulerHappensBeforeEdges(const AtomicContract &contract);
std::optional<std::int64_t> expectedAtomicSlot(llvm::StringRef role);

mlir::FailureOr<std::optional<AtomicContract>>
readAtomicContract(mlir::Operation *op);
mlir::LogicalResult verifyAtomicContractShape(mlir::Operation *op,
                                              const AtomicContract &contract);

mlir::Value atomicMemoryBase(mlir::Operation *op);
std::optional<std::int64_t> atomicSlotIndex(mlir::Operation *op);
mlir::Value stripThreadSafeView(mlir::Value value);
mlir::FailureOr<RetainPremiseInference>
inferRetainPremiseSource(mlir::Operation *op,
                         const AtomicContract &contract);
bool retainPremiseAllowsSource(RetainPremiseKind premise,
                               RetainPremiseSourceKind source);
mlir::LogicalResult verifyRetainPremise(mlir::Operation *op,
                                        const AtomicContract &contract);

} // namespace py::threadsafe
