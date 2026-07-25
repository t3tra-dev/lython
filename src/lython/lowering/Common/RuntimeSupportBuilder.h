#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/TargetParser/Triple.h"

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace py::runtime_library {

// Builds the native runtime support module directly from the compiler (the
// former hand-written `runtime/native/support.mlir`). The module is composed
// from high-level dialects (func, arith, math, cf, scf, memref, ub) and only
// drops to the `llvm` dialect for the irreducible Itanium C++ exception ABI.
// It is lowered to LLVM and linked into every compiled program by
// `linkEmbeddedNativeRuntime`.
//
// Almost all of it is target-independent; the triple is needed only by the OS
// cluster, whose errno accessor symbol and libc struct offsets have no
// portable spelling (see HostTargetLayout).
mlir::OwningOpRef<mlir::ModuleOp>
buildNativeRuntimeSupportModule(mlir::MLIRContext &context,
                                const llvm::Triple &triple);

} // namespace py::runtime_library
