#pragma once

// Type predicate for a container's {length, capacity} metadata view: a rank-1
// i64 memref whose slot 0 holds the element count. Kept next to the rest of
// the physical ABI mapping (not in each lower* TU) so every consumer shares
// one description of the layout.
//
// The view itself comes from RuntimeBundleLowerer::containerInteriorView --
// for a lane-carrying contract it is physicalValues()[1], for a handle-fronted
// one it is a borrowed view of the handle's length/capacity words. Readers
// must not assume which.

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

} // namespace py::lowering::collection_abi
