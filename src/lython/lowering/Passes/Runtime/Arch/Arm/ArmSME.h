#pragma once

#include "Common/RuntimeSupport.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include <memory>

namespace py::lowering::arch::arm {

bool usesSME(const py::TensorLoweringTarget &target);

void registerSMEDialects(mlir::DialectRegistry &registry);

// Splits the LHS transpose the SME kernel needs out of the contraction, while
// tensors are still values. Placed at the definition of A, an SSA value that
// nothing can write to afterwards, the transpose lands outside any loop that
// merely reads A -- so a loop-carried `x = A @ x` transposes once instead of
// once per iteration, and no alias analysis is involved.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMatmulLhsTransposePass(py::TensorLoweringTarget target = {});

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMatmulSMELoweringPass(py::TensorLoweringTarget target = {});

// Must run before createMatmulSMELoweringPass: that pass takes a matmul whole,
// so a block it should honour has to already be a loop around it.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMatmulSMEReductionBlockPass(py::TensorLoweringTarget target = {});

// Must run before the parallel chunker: it packs the rhs once per contraction,
// which is the whole point -- inside the chunk loop the copy would repeat per
// chunk.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMatmulSMERhsPackPass(py::TensorLoweringTarget target = {});

void addSMELinalgPipeline(mlir::OpPassManager &pipeline);

void addSMEPreControlFlowLLVMPrepPipeline(mlir::OpPassManager &pipeline);

void addSMEPostControlFlowLLVMPrepPipeline(mlir::OpPassManager &pipeline);

} // namespace py::lowering::arch::arm
