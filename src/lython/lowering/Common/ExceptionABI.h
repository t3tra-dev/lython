#pragma once

// The exception triple, spelled once.
//
// An exception travels between the compiled program and the EH runtime as
// three rank-1 memrefs: the exception object's header, its message's header,
// and the message's bytes. Two sides have to agree on that -- the lowering
// pass, which declares the runtime entry points at their call sites
// (Passes/Runtime/Ops/ExceptionOps.cpp), and the runtime support builder,
// which defines them (Common/TracebackSupportBuilder.cpp).
//
// ⛔ They agree by CALLING THIS, not by each spelling the types. Nothing
// downstream catches a drift -- checked, not assumed: `opt -passes=verify`
// accepts a four-argument call to a two-parameter definition and exits 0. The
// two sides are verified as MLIR separately and meet only as LLVM IR, where a
// disagreement links, runs, and reads its arguments off the wrong registers.
//
// The types used to be written out on both sides, and the definition's copy was
// a hand transcription of MLIR's memref descriptor layout -- a copy of
// something that can change without this tree noticing.

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace py::runtime_library {

inline llvm::SmallVector<mlir::Type, 3> exceptionTripleTypes(mlir::Builder &b) {
  return {mlir::MemRefType::get({3}, b.getI64Type()),
          mlir::MemRefType::get({2}, b.getI64Type()),
          mlir::MemRefType::get({mlir::ShapedType::kDynamic}, b.getI8Type())};
}

} // namespace py::runtime_library
