#include "PrimitiveTensorX86.h"

#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#include "mlir/Target/LLVMIR/Dialect/X86Vector/X86VectorToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"

namespace py::runtime_lowering::arch::x86 {

bool usesX86(const py::TensorLoweringTarget &target) {
  return target.usesX86();
}

bool usesSSE42(const py::TensorLoweringTarget &target) {
  return target.usesX86SSE42();
}

bool usesAVX2FMA(const py::TensorLoweringTarget &target) {
  return target.usesX86AVX2FMA();
}

void registerX86Dialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::x86vector::X86VectorDialect>();
}

void registerX86Translations(mlir::DialectRegistry &registry) {
  mlir::registerX86VectorDialectTranslation(registry);
}

void addX86LinalgPipeline(mlir::OpPassManager &pipeline) {
  // X86 SSE/AVX2 fast paths are expressed through the standard vector dialect
  // first. Architecture-specific matmul passes can be inserted here without
  // teaching the generic lowering path about x86 variants.
  pipeline.addPass(mlir::createCanonicalizerPass());
  pipeline.addPass(mlir::createCSEPass());
}

} // namespace py::runtime_lowering::arch::x86
