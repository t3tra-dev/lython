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

#include <cstdint>

namespace py::runtime_library {

inline llvm::SmallVector<mlir::Type, 3> exceptionTripleTypes(mlir::Builder &b) {
  return {mlir::MemRefType::get({3}, b.getI64Type()),
          mlir::MemRefType::get({2}, b.getI64Type()),
          mlir::MemRefType::get({mlir::ShapedType::kDynamic}, b.getI8Type())};
}

// The exception object's chain words (objects/exceptions.mlir,
// LyBaseException_New): `__cause__` and `__context__` are each one entity
// word -- the exception it names, 0 for None -- owning a reference, and
// `__suppress_context__` is 0 or 1. The EH runtime writes them at a raise and
// the lowering reads them for the attributes, so both take the offsets from
// here.
inline constexpr std::int64_t kExceptionCauseWord = 7;
inline constexpr std::int64_t kExceptionContextWord = 8;
inline constexpr std::int64_t kExceptionSuppressContextWord = 9;

// `LyEH_SetCause(raised, cause)` and `LyEH_SetCauseNone(raised)`: the object
// half of `raise X from Y` / `raise X from None`, each exception by its
// header.
inline mlir::Type exceptionHeaderType(mlir::Builder &b) {
  return mlir::MemRefType::get({3}, b.getI64Type());
}

} // namespace py::runtime_library
