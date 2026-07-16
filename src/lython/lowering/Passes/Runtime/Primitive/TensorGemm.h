#pragma once

#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

#include <memory>

namespace py::lowering {

inline constexpr llvm::StringLiteral kMatmulZeroInitAttr{
    "ly.prim_tensor.matmul_zero_init"};
inline constexpr llvm::StringLiteral kMatmulZeroInitFirstReductionAttr{
    "ly.prim_tensor.matmul_zero_init_first_reduction"};

// linalg.matmul carries its operand layout in `indexing_maps`: the default
// reads A as [M][K], but a pre-transposed LHS reads it as [K][M]. The two are
// the same op with the same operand count, so a pass that assumes the default
// would index a transposed A with the wrong extents and quietly compute
// garbage rather than fail. Every pass that hard-codes the default layout must
// gate on this.
bool hasDefaultMatmulMaps(mlir::linalg::MatmulOp matmul);

// True when the LHS is stored transposed, i.e. A is [K][M]: the layout the SME
// kernel wants, since it needs A's column k as a contiguous run.
bool hasTransposedLhsMatmulMaps(mlir::linalg::MatmulOp matmul);

// The register tile these passes agree on is target-dependent: it must fit the
// architecture's vector register file (see selectRegisterTile).
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMatmulZeroInitElisionPass(TensorLoweringTarget target = {});
// Cuts `matmul`'s reduction into `kc`-deep blocks, in place. `kc` must divide K:
// a partial block reaches the vector lowering as a non-identity layout map on
// the strided operands the parallel chunker cuts, which it rejects.
//
// The mechanism, not the policy: how deep to cut is a property of the kernel
// that will consume the blocks and of the machine it runs on, so the caller
// decides (see createMatmulSMEReductionBlockPass). What is generic is keeping
// the contraction's meaning across the cut -- in particular demoting a
// zero-init contract to zero-init-on-first-reduction, since C now accumulates
// across blocks.
mlir::LogicalResult tileMatmulReduction(mlir::linalg::MatmulOp matmul,
                                        int64_t kc,
                                        mlir::IRRewriter &rewriter);

// The divisor of `extent` nearest `target`, preferring the larger on a tie.
// `extent` itself is always a candidate, so a shape whose divisors all sit far
// from the target opts out.
int64_t divisorNearest(int64_t extent, int64_t target);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMatmulTilingPass(TensorLoweringTarget target = {});
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createMatmulPackingPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createPackedPanelHoistingPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createPackedPanelCopyHoistingPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createPackedPanelCopyVectorizationPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMatmulMicroTilingPass(TensorLoweringTarget target = {});
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMatmulVectorizationPass(TensorLoweringTarget target = {});

} // namespace py::lowering
