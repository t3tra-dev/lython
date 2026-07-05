#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace py::ownership {

inline constexpr llvm::StringLiteral kOwnedResultsAttr{
    "ly.ownership.owned_results"};
inline constexpr llvm::StringLiteral kOwnedResultContractsAttr{
    "ly.ownership.owned_result_contracts"};
inline constexpr llvm::StringLiteral kBorrowedResultsAttr{
    "ly.ownership.borrowed_results"};
inline constexpr llvm::StringLiteral kRetainArgsAttr{
    "ly.ownership.retain_args"};
inline constexpr llvm::StringLiteral kReleaseArgsAttr{
    "ly.ownership.release_args"};
inline constexpr llvm::StringLiteral kTransferArgsAttr{
    "ly.ownership.transfer_args"};
inline constexpr llvm::StringLiteral kObjectHeaderAttr{
    "ly.ownership.object_header"};
inline constexpr llvm::StringLiteral kOwnedLocalObjectAttr{
    "ly.ownership.owned_local_object"};
inline constexpr llvm::StringLiteral kObjectDeallocPartAttr{
    "ly.ownership.object_dealloc_part"};
inline constexpr llvm::StringLiteral kObjectReleaseToZeroAttr{
    "ly.ownership.object_release_to_zero"};
inline constexpr llvm::StringLiteral kAggregateRetainAttr{
    "ly.ownership.aggregate_retain"};
inline constexpr llvm::StringLiteral kAggregateReleaseAttr{
    "ly.ownership.aggregate_release"};

inline constexpr llvm::StringLiteral kAtomicRoleAttr{"ly.atomic.role"};
inline constexpr llvm::StringLiteral kAtomicOrderingAttr{"ly.atomic.ordering"};
inline constexpr llvm::StringLiteral kAtomicRetainPremiseAttr{
    "ly.atomic.retain_premise"};

inline constexpr llvm::StringLiteral kCallableTypeAttr{"callable_type"};

struct IndexSet {
  llvm::SmallVector<unsigned, 8> values;

  bool empty() const { return values.empty(); }
  bool contains(unsigned index) const;
};

struct FunctionContract {
  IndexSet ownedResults;
  llvm::SmallVector<std::string, 8> ownedResultContracts;
  IndexSet borrowedResults;
  IndexSet retainArgs;
  IndexSet releaseArgs;
  IndexSet transferArgs;
  bool objectReleaseToZero = false;

  bool hasAnyOwnershipAttr() const;
  bool consumesArg(unsigned index) const;
};

enum class AggregateOwnershipAction { Retain, Release };

struct AggregateOwnershipMarker {
  AggregateOwnershipAction action = AggregateOwnershipAction::Retain;
  std::string slot;
};

mlir::FailureOr<IndexSet>
parseIndexSetAttr(mlir::Operation *op, llvm::StringRef attrName,
                  std::optional<unsigned> upperBound = std::nullopt);

mlir::FailureOr<FunctionContract>
readFunctionContract(mlir::func::FuncOp function);
mlir::FailureOr<std::optional<AggregateOwnershipMarker>>
readAggregateOwnershipMarker(mlir::Operation *op);

bool isRuntimeManifestFunction(mlir::func::FuncOp function);
bool functionUsesOwnedReturnABI(mlir::func::FuncOp function);
bool functionOwnsResultAt(mlir::func::FuncOp function, unsigned resultIndex);
bool functionConsumesOperandAt(mlir::func::FuncOp function,
                               unsigned operandIndex);
bool functionReleasesOperandAt(mlir::func::FuncOp function,
                               unsigned operandIndex);
bool functionRetainsOperandAt(mlir::func::FuncOp function,
                              unsigned operandIndex);

struct RuntimeDeallocator {
  mlir::func::FuncOp function;
  std::string contractName;
  llvm::SmallVector<mlir::Type, 4> inputTypes;
  FunctionContract contract;
};

llvm::SmallVector<RuntimeDeallocator, 8>
collectRuntimeDeallocators(mlir::ModuleOp module);

bool valueRangeMatchesTypes(mlir::ValueRange values, unsigned offset,
                            llvm::ArrayRef<mlir::Type> types);
bool isObjectHeaderLikeType(mlir::Type type);
const RuntimeDeallocator *
findDeallocatorForValueGroup(mlir::ValueRange values, unsigned offset,
                             llvm::ArrayRef<RuntimeDeallocator> deallocators);
const RuntimeDeallocator *
findDeallocatorForValueGroup(mlir::ValueRange values, unsigned offset,
                             llvm::ArrayRef<RuntimeDeallocator> deallocators,
                             llvm::StringRef contractName);
llvm::SmallVector<mlir::Value, 4> valueSlice(mlir::ValueRange values,
                                             unsigned offset, unsigned size);
bool valueGroupEqualsEntryArgumentGroup(mlir::func::FuncOp function,
                                        llvm::ArrayRef<mlir::Value> group);
bool callResultGroupIsOwned(mlir::func::FuncOp callee, unsigned resultIndex);

enum class OwnershipKind { NonObject, Borrow, Own, Immortal };

llvm::StringRef ownershipKindName(OwnershipKind kind);
bool ownershipKindCarriesObjectResource(OwnershipKind kind);
OwnershipKind logicalOwnershipKind(mlir::Type logicalType, bool ownsObject);

struct OwnershipCondition {
  mlir::Value tag;
  std::int64_t activeTag = -1;
  unsigned memberCount = 0;
};

struct OwnershipConditionBranch {
  unsigned activeSuccessor = 0;
  unsigned inactiveSuccessor = 0;
};

std::optional<bool>
conditionTrueMeansActive(mlir::Value condition,
                         const OwnershipCondition &ownershipCondition);
std::optional<OwnershipConditionBranch>
classifyOwnershipConditionBranch(mlir::Operation *op,
                                 const OwnershipCondition &condition);

struct ResourceGroup {
  unsigned offset = 0;
  OwnershipKind ownership = OwnershipKind::Own;
  llvm::SmallVector<mlir::Value, 4> values;
  const RuntimeDeallocator *deallocator = nullptr;
  std::optional<OwnershipCondition> condition;
};

llvm::SmallVector<ResourceGroup, 8>
collectRuntimeResourceGroups(mlir::ValueRange values,
                             llvm::ArrayRef<RuntimeDeallocator> deallocators);
llvm::SmallVector<ResourceGroup, 8>
collectOwnedCallResultGroups(mlir::ModuleOp module, mlir::func::CallOp call,
                             llvm::ArrayRef<RuntimeDeallocator> deallocators);

class AliasAnalysis {
public:
  void build(mlir::Operation *root);
  void track(mlir::Value value);
  mlir::Value find(mlir::Value value);
  bool same(mlir::Value lhs, mlir::Value rhs);
  void unionValues(mlir::Value lhs, mlir::Value rhs);
  void aliasesOf(mlir::Value value,
                 llvm::SmallVectorImpl<mlir::Value> &aliases);

private:
  llvm::DenseMap<mlir::Value, mlir::Value> parent;
};

} // namespace py::ownership
