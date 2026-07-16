#ifndef LYTHON_LOWERING_PASSES_RUNTIME_ARCH_ARM_PRIMITIVETENSORAPPLEAMX_H
#define LYTHON_LOWERING_PASSES_RUNTIME_ARCH_ARM_PRIMITIVETENSORAPPLEAMX_H

#include "Common/RuntimeSupport.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace py::lowering::arch::apple {

bool usesAMX(const py::TensorLoweringTarget &target);

// Rewrites supported primitive f32 matmuls into an AMX kernel guarded by a
// run-time engine check, keeping the original linalg.matmul on the else path
// for machines that answer no. Runs before the generic tiling passes, which
// then specialize that fallback.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMatmulAMXLoweringPass(py::TensorLoweringTarget target = {});

// Synthesizes LyMatrixBackend: the cached run-time answer to "may this
// process issue AMX instructions". Runs on the final LLVM dialect module
// because it is spelled in libc calls and reserved instruction words.
mlir::LogicalResult materializeMatrixBackendProbe(mlir::ModuleOp module);

} // namespace py::lowering::arch::apple

#endif // LYTHON_LOWERING_PASSES_RUNTIME_ARCH_ARM_PRIMITIVETENSORAPPLEAMX_H
