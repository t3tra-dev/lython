#pragma once

// The one place `ly.ownership.owned_local_object` is written.
//
// It says so because it is checkable, not as a wish. Three call sites used to
// assemble this cast by hand, and every comment that reasoned about "the
// producers" got the count wrong: `ABI/EntityHeaderPrefix.h` justified
// `prefixIsInitializedAtDefinition` by describing two of them, and
// `Ops/GetItemOps.cpp` called itself the only one. Both conclusions happened
// to hold. Neither argument covered the code it was about.
//
// The shape matters beyond tidiness. `verifyInitialisationWindowIn` reads the
// marker as the model's `dup` and distinguishes it from `alloc` by the marker
// being a rooting cast rather than the op that creates the handle -- a fourth
// producer that marked, say, a `memref.view` would silently leave the gate's
// domain. One mint is what makes that distinction a property of the compiler
// instead of a property of three files agreeing.

#include "Ownership.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace py::lowering {

// An identity cast erased at reconciliation; the attributes are the whole
// content. The insertion point must already be set.
inline mlir::UnrealizedConversionCastOp
mintOwnedLocalMarker(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::ValueRange values, llvm::StringRef contract) {
  llvm::SmallVector<mlir::Type, 8> types;
  types.reserve(values.size());
  for (mlir::Value value : values)
    types.push_back(value.getType());
  auto marker =
      mlir::UnrealizedConversionCastOp::create(builder, loc, types, values);
  marker->setAttr(ownership::kOwnedLocalObjectAttr, builder.getUnitAttr());
  marker->setAttr(ownership::kOwnedLocalObjectContractAttr,
                  builder.getStringAttr(contract));
  return marker;
}

} // namespace py::lowering
