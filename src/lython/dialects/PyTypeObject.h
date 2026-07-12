#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace py {

class ClassOp;

namespace type_object {

inline constexpr llvm::StringLiteral kBaseException = "BaseException";
inline constexpr llvm::StringLiteral kException = "Exception";

ClassOp lookup(mlir::Operation *from, llvm::StringRef name);
mlir::FailureOr<llvm::SmallVector<llvm::StringRef, 8>>
mroNames(mlir::Operation *from, llvm::StringRef name);
mlir::LogicalResult verifyBases(ClassOp op);
mlir::FailureOr<bool> isSubclassOf(mlir::Operation *from,
                                   llvm::StringRef derived,
                                   llvm::StringRef base);
// Quiet subtype QUERY: false (no diagnostic) when either name is not a
// py.class symbol. Use from subtype lattice checks; isSubclassOf is for
// verification sites where an unknown name is an error.
bool isKnownSubclassOf(mlir::Operation *from, llvm::StringRef derived,
                       llvm::StringRef base);
mlir::FailureOr<bool> exceptionMatches(mlir::Operation *from,
                                       llvm::StringRef handler);

} // namespace type_object
} // namespace py
