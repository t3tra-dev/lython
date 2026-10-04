#include "ArmSME.h"

#include "mlir/Conversion/ArithToArmSME/ArithToArmSME.h"
#include "mlir/Conversion/ArmSMEToLLVM/ArmSMEToLLVM.h"
#include "mlir/Conversion/ArmSMEToSCF/ArmSMEToSCF.h"
#include "mlir/Conversion/VectorToArmSME/VectorToArmSME.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/ArmSME/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/Passes.h"

namespace py::lowering::arch::arm {

bool usesSME(const py::TensorLoweringTarget &target) {
  return target.usesArmSME();
}

void registerSMEDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arm_sme::ArmSMEDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::index::IndexDialect,
                  mlir::vector::VectorDialect>();
}

// The ArmSME claim itself: convert the vector form, legalize the shapes the
// tiles cannot take, fuse outer products, and enter streaming mode with ZA.
// Both entry points below run exactly this, and a pass added to one of them
// but not the other would make the two ArmSME paths disagree.
void addSMEVectorClaim(mlir::OpPassManager &pipeline) {
  pipeline.addPass(mlir::createConvertVectorToArmSMEPass());
  pipeline.addPass(mlir::arm_sme::createVectorLegalizationPass());
  pipeline.addNestedPass<mlir::func::FuncOp>(
      mlir::arm_sme::createOuterProductFusionPass());
  pipeline.addNestedPass<mlir::func::FuncOp>(
      mlir::arm_sme::createEnableArmStreamingPass(
          mlir::arm_sme::ArmStreamingMode::StreamingLocally,
          mlir::arm_sme::ArmZaMode::NewZA,
          /*ifRequiredByOps=*/true,
          /*ifContainsScalableVectors=*/false));
}

void addSMELinalgPipeline(mlir::OpPassManager &pipeline) {
  // Keep the high-level vector/outer-product form intact until the ArmSME
  // branch has a chance to claim it. The generic fallback is still responsible
  // for scalarizing unsupported vector shapes later.
  addSMEVectorClaim(pipeline);
  pipeline.addPass(mlir::createCanonicalizerPass());
  pipeline.addPass(mlir::createCSEPass());
}

void addSMEPreControlFlowLLVMPrepPipeline(mlir::OpPassManager &pipeline) {
  // Convert scalable vector outer-products to ArmSME while the vector form is
  // still intact. Tile stores are then expanded to SCF slice loops; ArmSME
  // tile allocation expects control-flow lowering before the final
  // ArmSME-to-LLVM conversion.
  pipeline.addPass(mlir::createArithToArmSMEConversionPass());
  addSMEVectorClaim(pipeline);
  pipeline.addPass(mlir::createConvertArmSMEToSCFPass());
  pipeline.addPass(mlir::createCanonicalizerPass());
  pipeline.addPass(mlir::createCSEPass());
}

namespace {

// convert-arm-sme-to-llvm, run on the functions that have something for it.
// It allocates tiles first, which builds a Liveness of the whole function,
// and everything it then rewrites is an ArmSME op or a value of a tile type
// (its own TODO says as much: "return early if the function contains no
// ArmSME ops").
//
// ⛔ Not run on every function: that is every runtime body and the whole
// module function of the program, each paying a liveness analysis for
// nothing -- 0.6 s of a 128-case batch.
class ConvertArmSMEToLLVMWhereUsed
    : public mlir::PassWrapper<ConvertArmSMEToLLVMWhereUsed,
                               mlir::OperationPass<mlir::func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertArmSMEToLLVMWhereUsed)

  llvm::StringRef getArgument() const final {
    return "lython-convert-arm-sme-to-llvm-where-used";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    pipeline.getDependentDialects(registry);
  }

  ConvertArmSMEToLLVMWhereUsed() : pipeline("func.func") {
    pipeline.addPass(mlir::createConvertArmSMEToLLVMPass());
  }
  ConvertArmSMEToLLVMWhereUsed(const ConvertArmSMEToLLVMWhereUsed &other)
      : mlir::PassWrapper<ConvertArmSMEToLLVMWhereUsed,
                          mlir::OperationPass<mlir::func::FuncOp>>(other),
        pipeline(other.pipeline) {}

  void runOnOperation() override {
    mlir::func::FuncOp function = getOperation();
    auto isTile = [](mlir::Type type) {
      return mlir::arm_sme::isValidSMETileVectorType(type);
    };
    bool used = function
                    .walk([&](mlir::Operation *op) {
                      if (mlir::isa<mlir::arm_sme::ArmSMEDialect>(
                              op->getDialect()) ||
                          llvm::any_of(op->getResultTypes(), isTile) ||
                          llvm::any_of(op->getOperandTypes(), isTile))
                        return mlir::WalkResult::interrupt();
                      for (mlir::Region &region : op->getRegions())
                        for (mlir::Block &block : region)
                          if (llvm::any_of(block.getArgumentTypes(), isTile))
                            return mlir::WalkResult::interrupt();
                      return mlir::WalkResult::advance();
                    })
                    .wasInterrupted();
    if (used && mlir::failed(runPipeline(pipeline, function)))
      signalPassFailure();
  }

private:
  mlir::OpPassManager pipeline;
};

} // namespace

void addSMEPostControlFlowLLVMPrepPipeline(mlir::OpPassManager &pipeline) {
  pipeline.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<ConvertArmSMEToLLVMWhereUsed>());
  pipeline.addPass(mlir::createCanonicalizerPass());
  pipeline.addPass(mlir::createCSEPass());
}

} // namespace py::lowering::arch::arm
