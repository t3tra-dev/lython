#pragma once

// Physical layout of evidence-collection length metadata: physicalValues()[1]
// is a rank-1 i64 memref whose slot 0 holds the element count. Kept next to
// the rest of the physical ABI mapping (not in each lower* TU) so every
// consumer shares one description of the layout.

#include "Runtime/Model/Bundles.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace py::lowering::collection_abi {

inline bool isCollectionMetaType(mlir::Type type) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memref || memref.getRank() != 1)
    return false;
  if (memref.hasStaticShape() && memref.getDimSize(0) < 1)
    return false;
  auto element = mlir::dyn_cast<mlir::IntegerType>(memref.getElementType());
  return element && element.getWidth() == 64;
}

inline mlir::FailureOr<mlir::Value>
loadCollectionLength(mlir::Operation *op, mlir::OpBuilder &builder,
                     const RuntimeBundle &bundle, llvm::StringRef label) {
  if (bundle.physicalValues().size() < 2)
    return op->emitError() << label
                           << " collection has no physical length metadata";
  mlir::Value meta = bundle.physicalValues()[1];
  if (!isCollectionMetaType(meta.getType()))
    return op->emitError() << label
                           << " collection length metadata has invalid type "
                           << meta.getType();
  mlir::Value slot =
      mlir::arith::ConstantIndexOp::create(builder, op->getLoc(), 0);
  return mlir::memref::LoadOp::create(builder, op->getLoc(), meta, slot)
      .getResult();
}

inline mlir::LogicalResult
touchCollectionEvidenceUse(mlir::Operation *op, mlir::OpBuilder &builder,
                           const RuntimeBundle &bundle, llvm::StringRef label) {
  mlir::FailureOr<mlir::Value> length =
      loadCollectionLength(op, builder, bundle, label);
  return mlir::failed(length) ? mlir::failure() : mlir::success();
}

} // namespace py::lowering::collection_abi
