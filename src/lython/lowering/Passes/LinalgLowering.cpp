// This pass lowers linalg ops to standard/scf while preserving only the known
// dialects that may survive until later lowering phases.

#include "Common/RuntimeSupport.h"

#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace py {
namespace {

struct LinalgLoweringPass
    : public mlir::PassWrapper<LinalgLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgLoweringPass)

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = module.getContext();
    mlir::RewritePatternSet patterns(ctx);
    mlir::linalg::populateLinalgToStandardConversionPatterns(patterns);

    mlir::ConversionTarget target(*ctx);
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalDialect<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                           mlir::bufferization::BufferizationDialect,
                           mlir::func::FuncDialect, mlir::tensor::TensorDialect,
                           mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
                           mlir::cf::ControlFlowDialect,
                           mlir::LLVM::LLVMDialect>();
    target.addIllegalDialect<mlir::linalg::LinalgDialect>();

    if (mlir::failed(
            applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLinalgLoweringPass() {
  return std::make_unique<LinalgLoweringPass>();
}

} // namespace py
