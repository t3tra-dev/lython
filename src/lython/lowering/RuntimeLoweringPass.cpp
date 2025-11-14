#include "RuntimeSupport.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
namespace {

struct RuntimeLoweringPass
    : public PassWrapper<RuntimeLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RuntimeLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    PyLLVMTypeConverter typeConverter(ctx);

    RewritePatternSet patterns(ctx);
    populatePyValueLoweringPatterns(typeConverter, patterns);
    populatePyTupleLoweringPatterns(typeConverter, patterns);
    populatePyDictLoweringPatterns(typeConverter, patterns);
    populatePyCallLoweringPatterns(typeConverter, patterns);

    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect>();
    target.addLegalOp<ModuleOp>();
    target.addIllegalOp<StrConstantOp, TupleEmptyOp, TupleCreateOp, DictEmptyOp,
                        DictInsertOp, NoneOp, FuncObjectOp, CallVectorOp,
                        CallOp>();

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createRuntimeLoweringPass() {
  return std::make_unique<RuntimeLoweringPass>();
}

static PassRegistration<RuntimeLoweringPass> pass;

} // namespace py
