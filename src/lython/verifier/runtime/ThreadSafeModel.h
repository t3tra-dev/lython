#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
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
AtomicOperationKind classifyAtomicOperationName(llvm::StringRef name);
std::optional<std::int64_t> constantIntValue(mlir::Value value);
AtomicRoleKind classifyAtomicRole(llvm::StringRef role,
                                  AtomicOperationKind operationKind);
AtomicOrderingRank requiredOrdering(AtomicRoleKind roleKind,
                                    AtomicOperationKind operationKind);
llvm::SmallVector<HappensBeforeEdge, 4>
schedulerHappensBeforeEdges(const AtomicContract &contract);
std::optional<std::int64_t> expectedAtomicSlot(llvm::StringRef role);

// Publish/acquire bookkeeping over scheduler happens-before edges — one rule
// for the live-op walk and the preserved-contract replay (the edge derivation
// itself lives in schedulerHappensBeforeEdges).
template <typename Tag> struct HappensBeforeLedger {
  llvm::StringSet<> published;
  llvm::StringMap<Tag> firstAcquire;

  void record(const AtomicContract &contract, Tag tag) {
    for (const HappensBeforeEdge &edge : schedulerHappensBeforeEdges(contract)) {
      if (edge.effect == HappensBeforeEffect::Publish) {
        published.insert(edge.resource);
        continue;
      }
      if (!firstAcquire.contains(edge.resource))
        firstAcquire[edge.resource] = tag;
    }
  }

  // First acquire whose resource never saw a publish; nullopt when balanced.
  std::optional<std::pair<llvm::StringRef, Tag>> firstUnmatchedAcquire() const {
    for (const auto &entry : firstAcquire)
      if (!published.contains(entry.getKey()))
        return std::make_pair(entry.getKey(), entry.getValue());
    return std::nullopt;
  }
};

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
