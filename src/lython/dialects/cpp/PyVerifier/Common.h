#ifndef LYTHON_PY_VERIFIER_COMMON_H
#define LYTHON_PY_VERIFIER_COMMON_H

#include "cpp/PyDialectTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

#include <optional>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {

enum class ThrowEffect {
  Unknown,
  NoThrow,
  MayThrow,
};

struct CallableEvidence {
  CallableType signature;
  CallableFuncOp target;
  ThrowEffect effect = ThrowEffect::MayThrow;
  mlir::ArrayAttr argNames;
  mlir::ArrayAttr kwonlyNames;
  MakeFunctionOp materializer;
  mlir::Value defaults;
  mlir::Value kwdefaults;
  mlir::Value closure;
  llvm::SmallVector<mlir::Type, 4> closureTypes;
  CallableFuncOp returnedCallable;
  mlir::FlatSymbolRefAttr returnedCallableSymbol;
  mlir::IntegerAttr returnedCallableDefaultsCount;
  mlir::ArrayAttr returnedCallableKwdefaultNames;
  llvm::StringSet<> defaultedKeywordNames;
  unsigned minPositionalCount = 0;
  bool hasKwdefaults = false;
};

struct SpecialMethodEvidence {
  CallableType signature;
  mlir::Operation *target = nullptr;
  ThrowEffect effect = ThrowEffect::MayThrow;
  mlir::ArrayAttr argNames;
  mlir::ArrayAttr kwonlyNames;
  llvm::SmallVector<mlir::Type, 4> closureTypes;
};

ClassOp lookupClassSymbol(mlir::Operation *from, ClassType classType);
CallableFuncOp lookupMethodByName(ClassOp classOp, mlir::StringRef methodName);
mlir::FailureOr<mlir::Type> lookupClassFieldType(mlir::Operation *from,
                                                 ClassType classType,
                                                 mlir::StringRef fieldName);

mlir::FailureOr<CallableEvidence> resolveCallableEvidence(mlir::Operation *op,
                                                          mlir::Value callable);

mlir::FailureOr<CallableEvidence>
resolveMakeFunctionEvidence(MakeFunctionOp op);

mlir::LogicalResult verifyMakeFunctionEvidence(MakeFunctionOp op);

mlir::FailureOr<TupleType>
requireTuple(mlir::Operation *op, mlir::Value operand, mlir::StringRef what);

mlir::LogicalResult verifyCallOperands(mlir::Operation *op,
                                       const CallableEvidence &evidence,
                                       mlir::Value posargs, mlir::Value kwnames,
                                       mlir::Value kwvalues);

mlir::FailureOr<SpecialMethodEvidence>
resolveSpecialMethodEvidence(mlir::Operation *op,
                             mlir::FlatSymbolRefAttr target,
                             mlir::Type calleeType, mlir::StringRef methodName);

mlir::LogicalResult verifySpecialMethodOperands(
    mlir::Operation *op, const SpecialMethodEvidence &evidence,
    mlir::ValueRange operands, std::optional<mlir::Type> resultType,
    mlir::StringRef methodName);

mlir::LogicalResult verifySpecialMethodOperandTypes(
    mlir::Operation *op, const SpecialMethodEvidence &evidence,
    mlir::TypeRange operandTypes, std::optional<mlir::Type> resultType,
    mlir::StringRef methodName);

mlir::LogicalResult
verifyUnaryMethodContract(mlir::Operation *op, mlir::Type calleeType,
                          mlir::Type receiverType, mlir::Type argumentType,
                          mlir::Type resultType, mlir::StringRef methodName);

mlir::LogicalResult verifyContainsMethodContract(mlir::Operation *op,
                                                 mlir::Type calleeType,
                                                 mlir::Type receiverType,
                                                 mlir::Type itemType);

} // namespace py

#endif // LYTHON_PY_VERIFIER_COMMON_H
