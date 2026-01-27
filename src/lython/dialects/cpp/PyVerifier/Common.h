#ifndef LYTHON_PY_VERIFIER_COMMON_H
#define LYTHON_PY_VERIFIER_COMMON_H

#include "cpp/PyDialectTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {

enum class ThrowEffect {
  Unknown,
  NoThrow,
  MayThrow,
};

ClassOp lookupClassSymbol(mlir::Operation *from, ClassType classType);
FuncOp lookupMethodByName(ClassOp classOp, mlir::StringRef methodName);

mlir::FailureOr<FuncSignatureType>
resolveCallableSignature(mlir::Operation *op, mlir::Value callable,
                         mlir::ArrayAttr &expectedKwNamesAttr);

mlir::FailureOr<ThrowEffect>
resolveCallableThrowEffect(mlir::Operation *op, mlir::Value callable);

mlir::FailureOr<TupleType> requireTuple(mlir::Operation *op,
                                        mlir::Value operand,
                                        mlir::StringRef what);

mlir::LogicalResult verifyVectorCallOperands(mlir::Operation *op,
                                             FuncSignatureType signature,
                                             mlir::Value posargs,
                                             mlir::Value kwnames,
                                             mlir::Value kwvalues,
                                             mlir::ArrayAttr expectedKwNamesAttr);

} // namespace py

#endif // LYTHON_PY_VERIFIER_COMMON_H
