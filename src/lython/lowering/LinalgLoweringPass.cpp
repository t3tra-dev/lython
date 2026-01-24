// This pass lowers linalg ops to standard/scf while allowing other dialects.

#include "RuntimeSupport.h"

#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace py {
namespace {

struct LinalgLoweringPass
    : public PassWrapper<LinalgLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgLoweringPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    RewritePatternSet patterns(ctx);
    linalg::populateLinalgToStandardConversionPatterns(patterns);

    ConversionTarget target(*ctx);
    target.addLegalOp<ModuleOp>();
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           tensor::TensorDialect, scf::SCFDialect,
                           memref::MemRefDialect, cf::ControlFlowDialect,
                           LLVM::LLVMDialect>();
    target.addIllegalDialect<linalg::LinalgDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createLinalgLoweringPass() {
  return std::make_unique<LinalgLoweringPass>();
}

} // namespace py
