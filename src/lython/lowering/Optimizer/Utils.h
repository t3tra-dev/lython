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

namespace runtime {

struct Call {
  static bool is(mlir::LLVM::CallOp op, llvm::StringRef callee);
};

struct Func {
  static mlir::LLVM::LLVMFuncOp getOrCreate(mlir::ModuleOp module,
                                            mlir::StringRef name,
                                            mlir::Type resultType,
                                            mlir::ValueRange operands);
  static bool eraseUnusedDecls(mlir::ModuleOp module);
};

} // namespace runtime

namespace value {

mlir::Value stripCasts(mlir::Value value);
mlir::Value stripPublications(mlir::Value value);

} // namespace value

namespace attr {

bool containsIndex(mlir::ArrayAttr attr, unsigned index);
mlir::ArrayAttr indexArray(mlir::MLIRContext *ctx,
                           const llvm::DenseSet<int> &indices);

} // namespace attr

namespace publication {

bool tracks(mlir::Type type);
bool result(mlir::Operation *funcLike, unsigned resultIndex);
int entryArg(CallableFuncOp func, mlir::Value value);
bool update(CallableFuncOp func, const llvm::DenseSet<int> &publishesArgs,
            const llvm::DenseSet<int> &capturesPublished,
            const llvm::DenseSet<int> &returnsPublished,
            const llvm::DenseSet<int> &readonlyArgs,
            const llvm::DenseSet<int> &mutableArgs);
void compute(mlir::ModuleOp module);
void insertBoundaries(mlir::ModuleOp module);
void prepare(mlir::ModuleOp module);

} // namespace publication

namespace call {

CallableFuncOp pyFunc(mlir::Operation *from, mlir::Value callable);
mlir::func::FuncOp localSelfHelper(mlir::func::FuncOp callee,
                                   mlir::ModuleOp module);
mlir::func::FuncOp freshInitHelper(mlir::func::FuncOp callee,
                                   mlir::ModuleOp module);
std::string publishedBorrowAttr(unsigned argIndex);
mlir::func::FuncOp publishedBorrowHelper(mlir::func::FuncOp callee,
                                         unsigned argIndex,
                                         mlir::ModuleOp module);
mlir::func::FuncOp preferredTarget(mlir::func::FuncOp callee,
                                   mlir::func::CallOp call,
                                   mlir::ModuleOp module);

template <typename CallbackT>
void forEachDirectPositionalOperand(CallOp op, CallbackT &&callback) {
  auto kwnames = value::stripCasts(op.getKwnames());
  auto kwvalues = value::stripCasts(op.getKwvalues());
  if (!mlir::isa_and_nonnull<TupleEmptyOp>(kwnames.getDefiningOp()) ||
      !mlir::isa_and_nonnull<TupleEmptyOp>(kwvalues.getDefiningOp()))
    return;

  mlir::Value posargs = value::stripCasts(op.getPosargs());
  if (auto tupleCreate = posargs.getDefiningOp<TupleCreateOp>())
    for (auto [idx, operand] : llvm::enumerate(tupleCreate.getOperands()))
      callback(static_cast<unsigned>(idx), operand);
}

template <typename CallbackT>
void forEachDirectPositionalOperand(InvokeOp op, CallbackT &&callback) {
  auto kwnames = value::stripCasts(op.getKwnames());
  auto kwvalues = value::stripCasts(op.getKwvalues());
  if (!mlir::isa_and_nonnull<TupleEmptyOp>(kwnames.getDefiningOp()) ||
      !mlir::isa_and_nonnull<TupleEmptyOp>(kwvalues.getDefiningOp()))
    return;

  mlir::Value posargs = value::stripCasts(op.getPosargs());
  if (auto tupleCreate = posargs.getDefiningOp<TupleCreateOp>())
    for (auto [idx, operand] : llvm::enumerate(tupleCreate.getOperands()))
      callback(static_cast<unsigned>(idx), operand);
}

void staticDefaults(mlir::ModuleOp module);
bool cleanupClassIncrefs(mlir::ModuleOp module);
void rewritePreferred(mlir::ModuleOp module);

} // namespace call

namespace class_state {

bool local(mlir::Value value);
bool fresh(mlir::Value value);
bool published(mlir::Value value);
void eliminatePublishes(mlir::ModuleOp module);
void proveLocalAccess(mlir::ModuleOp module);
void markFirstStores(mlir::ModuleOp module);

} // namespace class_state

namespace container {

bool cleanupDead(mlir::ModuleOp module);
void removeEmptyTuples(mlir::ModuleOp module);

} // namespace container

namespace consume {

void attrSetValues(mlir::ModuleOp module);
void listAppendValues(mlir::ModuleOp module);

} // namespace consume

namespace scalar {

void removeUnusedNone(mlir::ModuleOp module);
void dropNoneDecrefs(mlir::ModuleOp module);
void fuseStrConcat3(mlir::ModuleOp module);
void foldStaticBuiltinPrintRepr(mlir::ModuleOp module);
void foldIntConstants(mlir::ModuleOp module);
void hoistInts(mlir::ModuleOp module);
void cseConstants(mlir::ModuleOp module);
void dce(mlir::ModuleOp module);

} // namespace scalar

namespace int_fastpath {

void specialize(mlir::ModuleOp module);

} // namespace int_fastpath

namespace refcount {

void sinkClassDecrefs(mlir::ModuleOp module);
void markFinalLocal(mlir::ModuleOp module);

} // namespace refcount

namespace zero_cost {

void rewriteLocalAccess(mlir::ModuleOp module);

} // namespace zero_cost

namespace pipeline {

void classLayoutPre(mlir::ModuleOp module);
void callPre(mlir::ModuleOp module);
void containerPre(mlir::ModuleOp module);
void scalarPre(mlir::ModuleOp module);
void scalarPost(mlir::ModuleOp module);
void refcountPost(mlir::ModuleOp module);
void zeroCostProofPre(mlir::ModuleOp module);
void zeroCostRewritePre(mlir::ModuleOp module);
void preLowering(mlir::ModuleOp module);
void postValueLowering(mlir::ModuleOp module);
void finalLLVMCleanup(mlir::ModuleOp module);

} // namespace pipeline

} // namespace py::optimizer
