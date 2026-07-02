#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace py::runtime_lowering {

inline constexpr int64_t kPackedPanelAlignment = 64;
inline constexpr int64_t kPackedCopyVectorBits = 512;
inline constexpr llvm::StringLiteral kPackRhsCandidateAttr{
    "ly.prim_tensor.pack_rhs_candidate"};
inline constexpr llvm::StringLiteral kPackedRhsAttr{
    "ly.prim_tensor.packed_rhs"};
inline constexpr llvm::StringLiteral kPackedPanelAttr{
    "ly.prim_tensor.packed_panel"};
inline constexpr llvm::StringLiteral kPrepackedRhsAttr{
    "ly.prim_tensor.prepacked_rhs"};

struct RhsPrepackPlan {
  mlir::Value base;
  mlir::Value storage;
  mlir::Operation *anchor;
  int64_t totalK;
  int64_t totalN;
  int64_t panelK;
  int64_t panelN;
};

bool shouldPackRhsPanel(mlir::linalg::MatmulOp matmul);

bool tryPrepackFullRhsPanel(mlir::linalg::MatmulOp matmul,
                            mlir::IRRewriter &rewriter,
                            llvm::SmallVectorImpl<RhsPrepackPlan> &plans);

} // namespace py::runtime_lowering
