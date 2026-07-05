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

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMatmulSMELoweringPass();

void addSMELinalgPipeline(mlir::OpPassManager &pipeline);

void addSMEPreControlFlowLLVMPrepPipeline(mlir::OpPassManager &pipeline);

void addSMEPostControlFlowLLVMPrepPipeline(mlir::OpPassManager &pipeline);

} // namespace py::lowering::arch::arm
