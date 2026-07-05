#pragma once

#include "Common/RuntimeSupport.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"

namespace py::lowering::arch::x86 {

bool usesX86(const py::TensorLoweringTarget &target);

bool usesSSE42(const py::TensorLoweringTarget &target);

bool usesAVX2FMA(const py::TensorLoweringTarget &target);

void registerX86Dialects(mlir::DialectRegistry &registry);

void registerX86Translations(mlir::DialectRegistry &registry);

void addX86LinalgPipeline(mlir::OpPassManager &pipeline);

} // namespace py::lowering::arch::x86
