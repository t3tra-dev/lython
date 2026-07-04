#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

#include <memory>

namespace py::runtime_lowering {

inline constexpr llvm::StringLiteral kMatmulZeroInitAttr{
    "ly.prim_tensor.matmul_zero_init"};
inline constexpr llvm::StringLiteral kMatmulZeroInitFirstReductionAttr{
    "ly.prim_tensor.matmul_zero_init_first_reduction"};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMatmulZeroInitElisionPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createMatmulTilingPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createMatmulPackingPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createPackedPanelHoistingPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createPackedPanelCopyHoistingPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createPackedPanelCopyVectorizationPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMatmulMicroTilingPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMatmulVectorizationPass();

} // namespace py::runtime_lowering
