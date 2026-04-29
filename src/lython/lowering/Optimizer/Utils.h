#pragma once

#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"

#include <algorithm>
#include <string>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py::optimizer {

bool isCallTo(mlir::LLVM::CallOp callOp, llvm::StringRef calleeName);
mlir::LLVM::CallOp getOptionalDecRefUser(mlir::Value value);
void eraseCallAndOptionalDecRefUsers(mlir::LLVM::CallOp callOp);
mlir::LLVM::LLVMFuncOp getOrCreateRuntimeFunc(mlir::ModuleOp module,
                                              mlir::StringRef name,
                                              mlir::Type resultType,
                                              mlir::ValueRange operands);
mlir::Value materializeI64FromLong(mlir::ModuleOp module,
                                   mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::Value boxedLong);
mlir::Value stripIdentityCasts(mlir::Value value);
mlir::Value stripTransparentPublicationOps(mlir::Value value);
bool isPublicationSummaryResultType(mlir::Type type);
bool arrayAttrContainsIndex(mlir::ArrayAttr attr, unsigned index);
FuncOp resolveDirectPyFuncSymbol(mlir::Operation *from, mlir::Value callable);
bool funcResultIsPublished(mlir::Operation *funcLike, unsigned resultIndex);
int getEntryArgumentIndex(FuncOp func, mlir::Value value);
mlir::ArrayAttr buildSortedIndexArrayAttr(mlir::MLIRContext *ctx,
                                          const llvm::DenseSet<int> &indices);
bool updateFuncPublicationSummaryAttrs(
    FuncOp func, const llvm::DenseSet<int> &publishesArgs,
    const llvm::DenseSet<int> &capturesPublished,
    const llvm::DenseSet<int> &returnsPublished,
    const llvm::DenseSet<int> &readonlyArgs,
    const llvm::DenseSet<int> &mutableArgs);
bool isDefinitelyLocalStaticClassValue(mlir::Value value);
bool isDefinitelyFreshStaticClassValue(mlir::Value value);
bool isDefinitelyPublishedStaticClassValue(mlir::Value value);
mlir::func::FuncOp resolveLocalSelfHelper(mlir::func::FuncOp callee,
                                          mlir::ModuleOp module);
mlir::func::FuncOp resolveFreshInitHelper(mlir::func::FuncOp callee,
                                          mlir::ModuleOp module);
std::string getPublishedBorrowHelperAttrName(unsigned argIndex);
mlir::func::FuncOp resolvePublishedBorrowHelper(mlir::func::FuncOp callee,
                                                unsigned argIndex,
                                                mlir::ModuleOp module);
mlir::func::FuncOp resolvePreferredDirectCallTarget(mlir::func::FuncOp callee,
                                                    mlir::func::CallOp call,
                                                    mlir::ModuleOp module);

template <typename CallbackT>
void forEachDirectPositionalOperand(CallVectorOp op, CallbackT &&callback) {
  auto kwnames = stripIdentityCasts(op.getKwnames());
  auto kwvalues = stripIdentityCasts(op.getKwvalues());
  if (!mlir::isa_and_nonnull<TupleEmptyOp>(kwnames.getDefiningOp()) ||
      !mlir::isa_and_nonnull<TupleEmptyOp>(kwvalues.getDefiningOp()))
    return;

  mlir::Value posargs = stripIdentityCasts(op.getPosargs());
  if (auto tupleCreate = posargs.getDefiningOp<TupleCreateOp>())
    for (auto [idx, operand] : llvm::enumerate(tupleCreate.getOperands()))
      callback(static_cast<unsigned>(idx), operand);
}

template <typename CallbackT>
void forEachDirectPositionalOperand(CallOp op, CallbackT &&callback) {
  mlir::Value kwargs = stripIdentityCasts(op.getKwargs());
  if (!mlir::isa_and_nonnull<NoneOp>(kwargs.getDefiningOp()))
    return;

  mlir::Value posargs = stripIdentityCasts(op.getPosargs());
  if (auto tupleCreate = posargs.getDefiningOp<TupleCreateOp>())
    for (auto [idx, operand] : llvm::enumerate(tupleCreate.getOperands()))
      callback(static_cast<unsigned>(idx), operand);
}

template <typename CallbackT>
void forEachDirectPositionalOperand(InvokeOp op, CallbackT &&callback) {
  auto kwnames = stripIdentityCasts(op.getKwnames());
  auto kwvalues = stripIdentityCasts(op.getKwvalues());
  if (!mlir::isa_and_nonnull<TupleEmptyOp>(kwnames.getDefiningOp()) ||
      !mlir::isa_and_nonnull<TupleEmptyOp>(kwvalues.getDefiningOp()))
    return;

  mlir::Value posargs = stripIdentityCasts(op.getPosargs());
  if (auto tupleCreate = posargs.getDefiningOp<TupleCreateOp>())
    for (auto [idx, operand] : llvm::enumerate(tupleCreate.getOperands()))
      callback(static_cast<unsigned>(idx), operand);
}

void computeLocalPublicationSummaries(mlir::ModuleOp module);
void insertPublishesAtPublicationBoundaries(mlir::ModuleOp module);

bool cleanupDeadTuples(mlir::ModuleOp module);
void removeUnusedTupleEmpties(mlir::ModuleOp module);
void applyStaticMakeFunctionDefaults(mlir::ModuleOp module);
bool cleanupRedundantClassIncrefsAfterDirectCalls(mlir::ModuleOp module);
void rewriteDirectFuncCallsToPreferredHelpers(mlir::ModuleOp module);
void eliminateRedundantClassPublishes(mlir::ModuleOp module);
void markKnownLocalStaticClassAccesses(mlir::ModuleOp module);
void markConsumedAttrSetValues(mlir::ModuleOp module);
void markConsumedListAppendValues(mlir::ModuleOp module);
void markZeroInitializedSelfFirstStores(mlir::ModuleOp module);
void removeUnusedNoneOps(mlir::ModuleOp module);
void removeNoneDecrefs(mlir::ModuleOp module);
void hoistIntConstants(mlir::ModuleOp module);
void removeSmallIntDecrefs(mlir::ModuleOp module);
void cseStringCreation(mlir::ModuleOp module);
void cseSingletonGetters(mlir::ModuleOp module);
void eliminateBoolBoxingUnboxing(mlir::ModuleOp module);
void eliminateLongArithmeticRoundTrips(mlir::ModuleOp module);
void eliminateLongBoxingUnboxing(mlir::ModuleOp module);
void cseSmallIntFromI64(mlir::ModuleOp module);
void cseConstants(mlir::ModuleOp module);
void eliminateDeadCode(mlir::ModuleOp module);
void repairMissingDirectArgReturnIncRefs(mlir::ModuleOp module);
void removeDuplicateDecRefs(mlir::ModuleOp module);
void sinkClassDecrefsPastBorrowedAttrUses(mlir::ModuleOp module);
void markFinalLocalClassDecrefs(mlir::ModuleOp module);

void runClassLayoutPreLoweringOptimizations(mlir::ModuleOp module);
void runCallPreLoweringOptimizations(mlir::ModuleOp module);
void runContainerPreLoweringOptimizations(mlir::ModuleOp module);
void runScalarPreLoweringOptimizations(mlir::ModuleOp module);
void runScalarPostLoweringOptimizations(mlir::ModuleOp module);
void runRefcountPostLoweringOptimizations(mlir::ModuleOp module);

} // namespace py::optimizer
