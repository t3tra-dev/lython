#include "cpp/PyVerifier/Common.h"

namespace py {

mlir::LogicalResult CallOp::verify() {
  mlir::FailureOr<CallableEvidence> evidenceOr =
      resolveCallableEvidence(getOperation(), getCallable());
  if (mlir::failed(evidenceOr))
    return mlir::failure();
  const CallableEvidence &evidence = *evidenceOr;
  if (evidence.effect == ThrowEffect::MayThrow)
    return emitOpError("maythrow callee must be invoked with py.invoke");

  if (mlir::failed(verifyCallOperands(getOperation(), evidence, getPosargs(),
                                      getKwnames(), getKwvalues())))
    return mlir::failure();

  auto resultTypes = evidence.signature.getResultTypes();
  if (getNumResults() != resultTypes.size())
    return emitOpError("result count mismatch with callee signature");
  for (auto [result, expected] : llvm::zip(getResultTypes(), resultTypes)) {
    if (result != expected)
      return emitOpError("result types must match callee return types");
  }

  return mlir::success();
}

mlir::LogicalResult InvokeOp::verify() {
  mlir::FailureOr<CallableEvidence> evidenceOr =
      resolveCallableEvidence(getOperation(), getCallable());
  if (mlir::failed(evidenceOr))
    return mlir::failure();
  const CallableEvidence &evidence = *evidenceOr;
  if (evidence.effect == ThrowEffect::NoThrow)
    return emitOpError("nothrow callee must be invoked with py.call");

  if (mlir::failed(verifyCallOperands(getOperation(), evidence, getPosargs(),
                                      getKwnames(), getKwvalues())))
    return mlir::failure();

  auto resultTypes = evidence.signature.getResultTypes();
  bool allNone = llvm::all_of(
      resultTypes, [](mlir::Type type) { return isPyNoneType(type); });
  if (getNormalDestOperands().empty() && allNone) {
    // Allow dropping !py.none results for statement invokes.
  } else {
    if (getNormalDestOperands().size() != resultTypes.size())
      return emitOpError("normal destination operand count mismatch");
    for (auto [operand, expected] :
         llvm::zip(getNormalDestOperands(), resultTypes))
      if (operand.getType() != expected)
        return emitOpError(
            "normal destination operand types must match callee return types");
  }

  mlir::Block *normalDest = getNormalDest();
  if (normalDest->getNumArguments() != getNormalDestOperands().size())
    return emitOpError("normal destination block argument count mismatch");
  for (auto [operand, arg] :
       llvm::zip(getNormalDestOperands(), normalDest->getArguments()))
    if (operand.getType() != arg.getType())
      return emitOpError("normal destination block argument types must match");

  if (getUnwindDestOperands().size() != 1)
    return emitOpError("unwind destination must take a single !py.exception");
  if (!isPyExceptionType(getUnwindDestOperands()[0].getType()))
    return emitOpError("unwind destination must take !py.exception");
  mlir::Block *unwindDest = getUnwindDest();
  if (unwindDest->getNumArguments() != getUnwindDestOperands().size())
    return emitOpError("unwind destination block argument count mismatch");
  for (auto [operand, arg] :
       llvm::zip(getUnwindDestOperands(), unwindDest->getArguments()))
    if (operand.getType() != arg.getType())
      return emitOpError("unwind destination block argument types must match");
  return mlir::success();
}

} // namespace py
