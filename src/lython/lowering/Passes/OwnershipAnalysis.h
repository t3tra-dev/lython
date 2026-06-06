#pragma once

#include "mlir/IR/Region.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace py {

bool isPyOwnershipTrackedType(mlir::Type type);
bool isPyOwnershipImmortalOp(mlir::Operation *op);
bool isPyOwnershipIdentityTransform(mlir::Operation *op);
bool createsPyOwnedResult(mlir::Operation *op);
bool consumesPyOwnedOperand(mlir::Operation *op, mlir::Value operand);

/// Shared ownership alias analysis used by both refcount insertion and
/// ownership verification. Keeping this single implementation is important:
/// insertion and verification must agree on block-argument aliases introduced
/// by branches/invokes, otherwise the quantitative proof can become unsound.
class OwnershipAliasAnalysis {
public:
  using TypePredicate = llvm::function_ref<bool(mlir::Type)>;
  using ValuePredicate = llvm::function_ref<bool(mlir::Value)>;
  using IdentityPredicate = llvm::function_ref<bool(mlir::Operation *)>;

  OwnershipAliasAnalysis(mlir::Region &region, TypePredicate tracksType,
                         IdentityPredicate isIdentityTransform);
  OwnershipAliasAnalysis(mlir::Region &region, ValuePredicate tracksValue,
                         IdentityPredicate isIdentityTransform);

  mlir::Value getRoot(mlir::Value value) const;
  bool sameRoot(mlir::Value lhs, mlir::Value rhs) const;
  bool isCarrier(mlir::Value value) const;
  bool tracksThroughAlias(mlir::Value value) const;
  bool rootIsImmortal(mlir::Value root) const;
  bool rootHasAggregateBorrow(mlir::Value root) const;
  bool rootIsEntryBorrowed(mlir::Value value, mlir::Block &entry) const;
  void collectAliases(mlir::Value value,
                      llvm::SmallVectorImpl<mlir::Value> &aliases) const;

private:
  mlir::Value find(mlir::Value value) const;
  void unionSets(mlir::Value lhs, mlir::Value rhs);
  void unionSuccessorOperands(mlir::ValueRange operands, mlir::Block *successor,
                              TypePredicate tracksType);

  mutable llvm::DenseMap<mlir::Value, mlir::Value> parent;
  llvm::SmallPtrSet<mlir::Value, 16> carrierRoots;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value, 4>> members;
};

} // namespace py
