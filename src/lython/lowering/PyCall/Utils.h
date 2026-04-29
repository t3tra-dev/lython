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
void ensureLandingpad(mlir::Block *unwind, mlir::Location loc,
                      mlir::ConversionPatternRewriter &rewriter);
bool canUseVoidHelper(CallVectorOp op, mlir::func::FuncOp callee);
mlir::Value stripIdentityCasts(mlir::Value value);

mlir::FailureOr<mlir::LLVM::LLVMStructType>
getStaticClassObjectType(mlir::Operation *from, ClassType classType,
                         const PyLLVMTypeConverter &typeConverter);
mlir::Value createStaticClassSlot(mlir::Location loc,
                                  mlir::LLVM::LLVMStructType objectType,
                                  mlir::ConversionPatternRewriter &rewriter,
                                  mlir::Operation *anchor);

mlir::func::FuncOp resolvePreferredDirectHelper(mlir::func::FuncOp callee,
                                                mlir::ValueRange operands,
                                                mlir::ModuleOp module,
                                                bool allowVoidHelper);
void eraseNoneResultUsers(CallVectorOp op,
                          mlir::ConversionPatternRewriter &rewriter);
void materializeLogicalResults(mlir::Location loc, mlir::TypeRange logicalTypes,
                               mlir::ValueRange loweredResults,
                               llvm::SmallVectorImpl<mlir::Value> &results,
                               const PyLLVMTypeConverter &typeConverter,
                               mlir::ConversionPatternRewriter &rewriter);

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

void populatePyDirectCallLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                          mlir::RewritePatternSet &patterns);
void populatePyInvokeLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                      mlir::RewritePatternSet &patterns);

} // namespace py
