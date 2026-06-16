#include "cpp/PyVerifier/Common.h"

namespace py {
namespace {

mlir::Type iteratorElementType(mlir::Type type) {
  auto protocol = mlir::dyn_cast<ProtocolType>(type);
  if (!protocol || protocol.getProtocolName() != "Iterator" ||
      protocol.getArguments().size() != 1)
    return {};
  return protocol.getArguments().front();
}

mlir::FailureOr<SpecialMethodEvidence> specialEvidence(mlir::Operation *op,
                                                       llvm::StringRef name) {
  auto target = op->getAttrOfType<mlir::FlatSymbolRefAttr>("target");
  auto calleeTypeAttr = op->getAttrOfType<mlir::TypeAttr>("callee_type");
  mlir::Type calleeType =
      calleeTypeAttr ? calleeTypeAttr.getValue() : mlir::Type{};
  return resolveSpecialMethodEvidence(op, target, calleeType, name);
}

mlir::LogicalResult verifySpecialMethod(mlir::Operation *op,
                                        mlir::ValueRange operands,
                                        std::optional<mlir::Type> resultType,
                                        llvm::StringRef name) {
  mlir::FailureOr<SpecialMethodEvidence> evidence = specialEvidence(op, name);
  if (mlir::failed(evidence))
    return mlir::failure();
  return verifySpecialMethodOperands(op, *evidence, operands, resultType, name);
}

mlir::LogicalResult
verifySpecialMethodTypes(mlir::Operation *op, mlir::TypeRange operandTypes,
                         std::optional<mlir::Type> resultType,
                         llvm::StringRef name) {
  mlir::FailureOr<SpecialMethodEvidence> evidence = specialEvidence(op, name);
  if (mlir::failed(evidence))
    return mlir::failure();
  return verifySpecialMethodOperandTypes(op, *evidence, operandTypes,
                                         resultType, name);
}

} // namespace

mlir::LogicalResult IterOp::verify() {
  mlir::Type resultElement = iteratorElementType(getResult().getType());
  bool returnedSelf = getOperation()->hasAttr("returned_self");
  std::optional<mlir::Type> expectedResult =
      returnedSelf ? std::optional<mlir::Type>{} : getResult().getType();
  if (mlir::failed(verifySpecialMethod(getOperation(), {getIterable()},
                                       expectedResult, "__iter__")))
    return mlir::failure();
  if (returnedSelf && getResult().getType() != getIterable().getType())
    return emitOpError("returned_self requires result type to match iterable "
                       "type");
  // Iterable conformance is established by the frontend protocol oracle;
  // the IR-level check is that an iterator of an iterator is the identity
  // element-wise.
  if (resultElement) {
    if (mlir::Type inputElement =
            iteratorElementType(getIterable().getType())) {
      if (inputElement != resultElement)
        return emitOpError("iterating an iterator must preserve the element "
                           "type");
    }
  }
  return mlir::success();
}

mlir::LogicalResult NextOp::verify() {
  mlir::Type elementType = iteratorElementType(getIterator().getType());
  if (mlir::failed(verifySpecialMethod(getOperation(), {getIterator()},
                                       getElement().getType(), "__next__")))
    return mlir::failure();
  if (elementType && getElement().getType() != elementType)
    return emitOpError("element result type ")
           << getElement().getType() << " does not match iterator element "
           << elementType;
  if (getNext().getType() != getIterator().getType())
    return emitOpError("advanced iterator must keep the iterator type");
  return mlir::success();
}

mlir::LogicalResult EnterOp::verify() {
  if (getOperation()->hasAttr("async_method"))
    return emitOpError("cannot reference an async method");
  return verifySpecialMethod(getOperation(), {getManager()},
                             getResult().getType(), "__enter__");
}

mlir::LogicalResult ExitOp::verify() {
  if (getOperation()->hasAttr("async_method"))
    return emitOpError("cannot reference an async method");
  return verifySpecialMethod(
      getOperation(),
      {getManager(), getExcType(), getExcValue(), getTraceback()},
      getResult().getType(), "__exit__");
}

mlir::LogicalResult AEnterOp::verify() {
  std::optional<mlir::Type> expectedResult;
  if (!getOperation()->hasAttr("async_method"))
    expectedResult = getResult().getType();
  return verifySpecialMethod(getOperation(), {getManager()}, expectedResult,
                             "__aenter__");
}

mlir::LogicalResult AExitOp::verify() {
  std::optional<mlir::Type> expectedResult;
  if (!getOperation()->hasAttr("async_method"))
    expectedResult = getResult().getType();
  return verifySpecialMethod(
      getOperation(),
      {getManager(), getExcType(), getExcValue(), getTraceback()},
      expectedResult, "__aexit__");
}

mlir::LogicalResult SendOp::verify() {
  return verifySpecialMethod(getOperation(), {getReceiver(), getValue()},
                             getResult().getType(), "send");
}

mlir::LogicalResult ThrowOp::verify() {
  auto args = mlir::dyn_cast<TupleType>(getArgs().getType());
  if (!args)
    return emitOpError("args operand must be !py.tuple");
  llvm::SmallVector<mlir::Type, 4> operandTypes{getReceiver().getType()};
  operandTypes.append(args.getElementTypes().begin(),
                      args.getElementTypes().end());
  return verifySpecialMethodTypes(getOperation(), operandTypes,
                                  getResult().getType(), "throw");
}

mlir::LogicalResult CloseOp::verify() {
  return verifySpecialMethod(getOperation(), {getReceiver()},
                             getResult().getType(), "close");
}

mlir::LogicalResult ASendOp::verify() {
  std::optional<mlir::Type> expectedResult;
  if (!getOperation()->hasAttr("async_method"))
    expectedResult = getResult().getType();
  return verifySpecialMethod(getOperation(), {getReceiver(), getValue()},
                             expectedResult, "asend");
}

mlir::LogicalResult AThrowOp::verify() {
  std::optional<mlir::Type> expectedResult;
  if (!getOperation()->hasAttr("async_method"))
    expectedResult = getResult().getType();
  auto args = mlir::dyn_cast<TupleType>(getArgs().getType());
  if (!args)
    return emitOpError("args operand must be !py.tuple");
  llvm::SmallVector<mlir::Type, 4> operandTypes{getReceiver().getType()};
  operandTypes.append(args.getElementTypes().begin(),
                      args.getElementTypes().end());
  return verifySpecialMethodTypes(getOperation(), operandTypes, expectedResult,
                                  "athrow");
}

mlir::LogicalResult ACloseOp::verify() {
  std::optional<mlir::Type> expectedResult;
  if (!getOperation()->hasAttr("async_method"))
    expectedResult = getResult().getType();
  return verifySpecialMethod(getOperation(), {getReceiver()}, expectedResult,
                             "aclose");
}

mlir::LogicalResult LenOp::verify() {
  if (!mlir::isa<IntType>(getResult().getType()))
    return emitOpError("result must be !py.int");
  auto calleeTypeAttr =
      getOperation()->getAttrOfType<mlir::TypeAttr>("callee_type");
  CallableType signature = calleeTypeAttr
                               ? getCallableContract(calleeTypeAttr.getValue())
                               : CallableType{};
  if (!signature)
    return emitOpError("callee_type must be Callable");
  if (signature.getPositionalTypes().size() != 1)
    return emitOpError("__len__ contract must take exactly the receiver");
  if (!signature.getKwOnlyTypes().empty() || signature.hasVararg() ||
      signature.hasKwarg())
    return emitOpError("__len__ contract must not take keyword, vararg, or "
                       "kwarg parameters");
  if (!isSubtypeOf(getInput().getType(), signature.getPositionalTypes().front(),
                   getOperation()))
    return emitOpError("input type ")
           << getInput().getType() << " does not satisfy __len__ receiver type "
           << signature.getPositionalTypes().front();
  if (signature.getResultTypes().size() != 1 ||
      !mlir::isa<IntType>(signature.getResultTypes().front()))
    return emitOpError("__len__ contract must return !py.int");
  return mlir::success();
}

} // namespace py
