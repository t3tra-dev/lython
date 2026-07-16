#include "X86.h"

#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#include "mlir/Transforms/Passes.h"

namespace py::lowering::arch::x86 {

bool usesX86(const py::TensorLoweringTarget &target) {
  return target.usesX86();
}

void registerX86Dialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::x86vector::X86VectorDialect>();
}

void registerX86Translations(mlir::DialectRegistry &registry) {
  (void)registry;
}

void addX86LinalgPipeline(mlir::OpPassManager &pipeline) {
  // Nothing x86-specific yet: SSE/AVX2 reach the vector dialect through the
  // generic path, and the one place the target's shape shows through -- the
  // register tile, which SSE4.2's 16-register file cannot hold at the default
  // width -- is solved there from the file's capacity rather than from the ISA
  // (see selectRegisterTile). A pass that only x86 can run belongs here.
  pipeline.addPass(mlir::createCanonicalizerPass());
  pipeline.addPass(mlir::createCSEPass());
}

} // namespace py::lowering::arch::x86
