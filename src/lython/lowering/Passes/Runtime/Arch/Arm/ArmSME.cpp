#include "ArmSME.h"

#include "mlir/Conversion/ArithToArmSME/ArithToArmSME.h"
#include "mlir/Conversion/ArmSMEToLLVM/ArmSMEToLLVM.h"
#include "mlir/Conversion/ArmSMEToSCF/ArmSMEToSCF.h"
#include "mlir/Conversion/VectorToArmSME/VectorToArmSME.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
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

void addSMEPostControlFlowLLVMPrepPipeline(mlir::OpPassManager &pipeline) {
  pipeline.addNestedPass<mlir::func::FuncOp>(
      mlir::createConvertArmSMEToLLVMPass());
  pipeline.addPass(mlir::createCanonicalizerPass());
  pipeline.addPass(mlir::createCSEPass());
}

} // namespace py::lowering::arch::arm
