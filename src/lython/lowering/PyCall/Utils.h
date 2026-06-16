#pragma once

#include "Common/LoweringUtils.h"
#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Location.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {

void emitTracebackPush(mlir::Location loc, mlir::func::FuncOp func,
                       RuntimeAPI &runtime,
                       mlir::ConversionPatternRewriter &rewriter);
bool isBuiltinPrintCallable(mlir::Value callable);
bool isBuiltinPrintRawCallable(mlir::Value callable);
void ensureLandingpad(mlir::Block *unwind, mlir::Location loc,
                      mlir::ConversionPatternRewriter &rewriter);
bool canUseVoidHelper(CallOp op, mlir::func::FuncOp callee);
mlir::Value stripBridgeCasts(mlir::Value value);

mlir::func::FuncOp resolvePreferredDirectHelper(mlir::func::FuncOp callee,
                                                mlir::ValueRange operands,
                                                mlir::ModuleOp module,
                                                bool allowVoidHelper);
mlir::LogicalResult appendFlattenedCallOperands(
    mlir::Location loc, mlir::ValueRange elements, mlir::FunctionType funcType,
    unsigned directInputCount, llvm::SmallVectorImpl<mlir::Value> &operands,
    mlir::RewriterBase &rewriter, const PyLLVMTypeConverter &typeConverter);
void eraseNoneResultUsers(CallOp op, mlir::RewriterBase &rewriter);
void materializeLogicalResults(mlir::Location loc, mlir::TypeRange logicalTypes,
                               mlir::ValueRange loweredResults,
                               llvm::SmallVectorImpl<mlir::Value> &results,
                               const PyLLVMTypeConverter &typeConverter,
                               mlir::RewriterBase &rewriter);

void materializeInvokeNormalResult(InvokeOp op, mlir::Value loweredResult,
                                   mlir::ConversionPatternRewriter &rewriter);
mlir::Block *
createInvokeNormalBridge(mlir::Block *finalDest, mlir::Type bridgeArgType,
                         mlir::Location loc,
                         mlir::ConversionPatternRewriter &rewriter);
void finalizeInvokeNormalBridge(mlir::Block *bridge, mlir::Block *finalDest,
                                mlir::Value forwardedValue, mlir::Location loc,
                                mlir::ConversionPatternRewriter &rewriter);
void eraseInvokeNormalSeedDrops(InvokeOp op, mlir::Value logicalSeed,
                                mlir::ConversionPatternRewriter &rewriter);

namespace lowering::call::direct::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace lowering::call::direct::Patterns

namespace lowering::call::invoke::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace lowering::call::invoke::Patterns

} // namespace py
